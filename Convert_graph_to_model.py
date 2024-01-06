# -*- coding:utf-8 -*-
import graph
import os
import re

# 如果某节点在这个白名单之中，则不需要在 init 函数之中进行处理
declaration_whitelist = [
    "hidden",
    "__getitem__",
    "Add",
    "Mul",
    "Concat",
    "Input",
    "Size",
    "View",
]

# 找出这个子图的输入，它们是子图中节点的前序，而不是子图中的节点。
def get_input_names(subgraph, full_graph, check_stage=True):
    nodes = subgraph.nodes
    input_names = {}
    counter = 0
    for node_id in nodes:
        if (node_id in full_graph.in_edges and len(full_graph.in_edges[node_id]) > 0):
            for in_node in full_graph.in_edges[node_id]:
                if in_node.stage_id != nodes[node_id].stage_id and check_stage:
                    if full_graph.nodes[node_id].node_desc.startswith("hidden"):  # transformer中有
                        continue
                    input_names[in_node.node_id] = "input%d" % counter
                    counter += 1
        else:
            if subgraph.nodes[node_id].node_desc.startswith("Input"):
                input_names[node_id] = "input%d" % counter
                counter += 1
    return input_names


# 计算出该子图的输出，即子图中具有子图外的边的节点
def get_output_names(subgraph, full_graph, counter):
    nodes = subgraph.nodes
    output_names = {}
    for node_id in nodes:
        if (node_id in full_graph.edges and
                len(full_graph.edges[node_id]) > 0):
            for out_node in full_graph.edges[node_id]:
                if out_node.stage_id != nodes[node_id].stage_id:
                    if full_graph.nodes[node_id].node_desc.startswith("hidden"):  # transformer中有
                        continue
                    output_names[node_id] = "out%d" % counter
                    counter += 1
        else:
            output_names[node_id] = "out%d" % counter
            counter += 1
    return output_names, counter


def get_tensor_names_list(graph_output_names):
    return [graph_output_names[node_id] for node_id in sorted(graph_output_names.keys())]


def get_output_tuple_str(output_names_list):
    if len(output_names_list) == 1:
        return output_names_list[0]
    return "(%s)" % ", ".join(output_names_list)


def convert_subgraph_to_model(subgraph, full_graph, num_subgraphs, module_name,
                              initialize_weights, model_template_filename, output_filename):
    model_template = open(model_template_filename, 'r').read()
    nodes = subgraph.topological_sort()
    import_statements = []
    module_methods = []

    counter = 0
    layer_names = {}
    layer_names_and_declarations = []
    function_definition = []
    input_names = get_input_names(subgraph, full_graph)
    num_inputs = len(input_names)
    output_names = input_names.copy()
    sources = subgraph.sources()

    # 构建forward函数定义部分，为后续生成代码做准备
    for node_id in input_names:
        output_name = "out%d" % counter
        function_definition.append("%s = %s.clone()" % (output_name, input_names[node_id]))
        output_names[node_id] = output_name
        counter += 1

    for node in nodes:
        layer_call = None  # 层相关信息
        layer_name = "self.layer%d" % counter
        output_name = "out%d" % counter
        layer_declaration = "torch.nn.%s" % (
            node.node_desc.replace("inplace", "inplace=True"))  # Relu
        layer_names[node.node_id] = layer_name
        if node.node_id not in output_names:
            output_names[node.node_id] = output_name

        # 跳过不需要声明的层
        # for declaration in declaration_specialcase:
        #     if node.node_desc.startswith(declaration):
        #         break

        # 归并import语句
        import_statements = list(set(import_statements))
        # 如果节点描述不在声明白名单之中，则处理
        found = False
        for declaration in declaration_whitelist:
            if node.node_desc.startswith(declaration):
                found = True
        if not found:
            layer_names_and_declarations.append((layer_name, layer_declaration))

        if node.node_id in full_graph.in_edges:
            in_edges = full_graph.in_edges[node.node_id]
        else:
            in_edges = []
        if len(in_edges) == 0 and node.node_desc.startswith("Input"):
            pass
        else:  # 看看节点是否在内置运算符之中
            if node.node_desc.startswith("Size"):
                assert (len(in_edges) == 1)
                m = re.search(r'Size\((-?\d+)\)', node.node_desc)
                idx = int(m.group(1))  # 正则表达式匹配到的第一组子串
                layer_call = "%s = %s.size(%d)" % (output_name,
                                                   output_names[in_edges[0].node_id],
                                                   idx)
            elif node.node_desc.startswith("View"):
                size_node_ids = []
                input_node_id = None
                for i in range(len(in_edges)):
                    if in_edges[i].node_desc.startswith("Size"):
                        size_node_id = in_edges[i].node_id
                        size_node_ids.append(size_node_id)
                    else:
                        input_node_id = in_edges[i].node_id
                m = re.search(r'View\((-?\d+)\)', node.node_desc)
                if m is None:
                    size_output_names = [output_names[size_node_id] for size_node_id in size_node_ids]
                    layer_call = "%s = %s.view(%s)" % (output_name,
                                                       output_names[input_node_id],
                                                       ', '.join(size_output_names))
                else:
                    size = int(m.group(1))
                    layer_call = "%s = %s.view(%s, %d)" % (output_name,
                                                           output_names[input_node_id],
                                                           output_names[size_node_id],
                                                           size)
            elif node.node_desc.startswith("Add"):
                assert (len(in_edges) == 2)
                node1 = in_edges[0]
                node2 = in_edges[1]
                if len(full_graph.edges[node1.node_id]) > 1:
                    tmp = node1
                    node1 = node2
                    node2 = tmp
                layer_call = "%s = %s + %s" % (output_names[node1.node_id],
                                               output_names[node1.node_id],
                                               output_names[node2.node_id])
                output_names[node.node_id] = output_names[node1.node_id]
            elif node.node_desc.startswith("Mul"):
                assert (len(in_edges) == 2)
                node1 = in_edges[0]
                node2 = in_edges[1]
                if len(full_graph.edges[node1.node_id]) > 1:
                    tmp = node1
                    node1 = node2
                    node2 = tmp
                layer_call = "%s = %s * %s" % (output_names[node1.node_id],
                                               output_names[node1.node_id],
                                               output_names[node2.node_id])
                output_names[node.node_id] = output_names[node1.node_id]
            elif node.node_desc.startswith("Concat"):
                m = re.search(r'Concat\((-?\d+)\)', node.node_desc)
                dim = int(m.group(1))
                layer_call = "%s = torch.cat([%s], %d)" % (
                    output_name,
                    ", ".join([output_names[in_node.node_id]
                               for in_node in in_edges]), dim)
            elif node.node_desc.startswith("hidden"):
                pass
            else:  # 如果不是内置运算，就直接设置，这里为 'out2 = self.layer2(out0, out1)'
                layer_call = "%s = %s(%s)" % (output_name, layer_name,
                                              ", ".join([output_names[in_node.node_id]
                                                         for in_node in in_edges]))
        if layer_call is not None:
            function_definition.append(layer_call)
        counter += 1

    # 确保模块输出是按照原始模型的顺序输出
    full_graph.populate_depths()
    graph_output_names, _ = get_output_names(subgraph, full_graph, 0)
    for key in graph_output_names:
        graph_output_names[key] = output_names[key]
    output_names_list = get_tensor_names_list(graph_output_names)
    num_outputs = len(output_names_list)
    function_definition.append("return %s" %
                               get_output_tuple_str(output_names_list))

    # 层声明被添加到模块的构造函数中。 函数定义被添加到模块的' forward()'方法中。
    layer_declarations_str = "\n        ".join(["%s = %s" % (x[0], x[1]) for x in layer_names_and_declarations])
    # 如果需要初始化权重，则做处理
    if initialize_weights:
        layer_declarations_str += "\n        self._initialize_weights()"
        module_methods.append("""def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)""")
    function_definition_str = "\n        ".join(function_definition)
    input_names_list = get_tensor_names_list(input_names)
    input_names = ", ".join(input_names_list)
    # 应用模版文件生成模型
    # 使用转换过程中生成的python语句对模版文件进行填充
    model = model_template % {"layer_declarations": layer_declarations_str,
                              "function_definition": function_definition_str,
                              "module_name": module_name,
                              "inputs": input_names,
                              "import_statements": "\n".join(import_statements),
                              "module_methods": "\n\n".join(module_methods)}
    with open(output_filename, 'w') as f:
        f.write(model)
    return num_inputs, num_outputs


def fuse_subgraphs_to_module(full_graph, subgraphs, model_name,
                             initialize_weights, model_template_filename,
                             output_filename):
    # PyTorch模块是所生成阶段的名称(类型为torch.nn.Module)。
    # Python模块是对包含这些生成的torch.nn.Modules的文件名的命名。
    module_methods = []

    # 加载模板
    model_template = open(model_template_filename, 'r').read()
    # 归并模块名称
    pytorch_modules = []
    python_modules = []
    for i in range(len(subgraphs)):
        pytorch_modules.append("Stage%d" % i)
        python_modules.append("stage%d" % i)

    # 处理函数定义和层定义
    layer_declarations = []
    function_definition = []
    for i, pytorch_module in enumerate(pytorch_modules):
        layer_declarations.append("self.stage%d = %s()" % (i, pytorch_module))
    if initialize_weights:
        layer_declarations.append("self._initialize_weights()")
        module_methods.append("""def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)""")

    output_counter = 0
    output_names = {}
    graph_input_names = get_input_names(full_graph, full_graph, check_stage=False)
    for key in graph_input_names:
        output_names[key] = graph_input_names[key]
    subgraph_inputs = []
    subgraph_outputs = []
    # 遍历子图，构建输出和输入
    for i, subgraph in enumerate(subgraphs):
        subgraph_input_names = get_input_names(subgraph, full_graph)
        subgraph_output_names, output_counter = get_output_names(
            subgraph, full_graph, output_counter)
        for key in subgraph_input_names:
            subgraph_input_names[key] = output_names[key]
        for key in subgraph_output_names:
            output_names[key] = subgraph_output_names[key]

        function_definition.append("%s = self.stage%d(%s)" % (
            get_output_tuple_str(get_tensor_names_list(subgraph_output_names)),
            i, ", ".join(get_tensor_names_list(subgraph_input_names))))
        subgraph_inputs.append(get_tensor_names_list(subgraph_input_names))
        subgraph_outputs.append(get_tensor_names_list(subgraph_output_names))

    # 添加输出信息
    function_definition.append("return %s" %
                               get_output_tuple_str(get_tensor_names_list(subgraph_output_names)))
    function_definition_str = "\n        ".join(function_definition)
    # 添加import信息
    import_statements = ["from .%s import %s" % (python_module, pytorch_module)
                         for (python_module, pytorch_module) in zip(python_modules, pytorch_modules)]
    input_names = get_input_names(full_graph, full_graph, check_stage=False)
    input_names = ", ".join(get_tensor_names_list(input_names))
    # 应用模版文件
    model = model_template % {"layer_declarations": "\n        ".join(layer_declarations),
                              "function_definition": function_definition_str,
                              "module_name": model_name,
                              "inputs": input_names,
                              "import_statements": "\n".join(import_statements),
                              "module_methods": "\n\n".join(module_methods)}

    print("Done with sub-graph fusion...")
    # 输出文件
    with open(output_filename, 'w') as f:
        f.write(model)
    return python_modules, pytorch_modules, subgraph_inputs, subgraph_outputs


if __name__ == '__main__':
    profile_name = '../DP_Partition/Partition_result/densenet121/gpu=8.txt'
    model_template_filename = "templates/model.py.template"  # 模版文件,生成模型
    init_template_filename = "templates/__init__.py.template"
    conf_template_filename = "templates/conf.json.template"
    # stage_to_num_ranks_map = "0:1, 1:1, 2:1, 3:1"
    model_name = "densenet121"
    arch = "densenet121"
    output_directory = '../DP_Partition/Partition_modules/densenet121_8/'

    # mkdir output_directory
    # subprocess.check_output("mkdir -p %s" % output_directory, shell=True)

    # 从graph文件中加载，得到一个图
    input_node = graph.Node("input_node", node_desc="Input")
    full_graph = graph.Gragh.from_str(open(profile_name, 'r').read())
    initialize_weights = "resnet18"
    input_node.stage_id = 0
    sinks = full_graph.sinks()
    for sink in sinks:  # node 71
        if sink.node_desc.startswith("__getitem__"):
            full_graph.remove_node(sink)

    # 分割为一系列子图
    subgraphs = full_graph.partition_graph()

    # 遍历每一个子图
    for i, subgraph in enumerate(subgraphs):
        module_name = "Stage%d" % i
        module_filename = "stage%d.py" % i
        # 把这个子图转换成一个module
        num_inputs, num_outputs = convert_subgraph_to_model(subgraph, full_graph, len(subgraphs),
                                                            module_name, initialize_weights,
                                                            model_template_filename,
                                                            os.path.join(output_directory, module_filename))
        print("Done generating %s..." % module_filename)

    # 把这些子图合并成一个大图
    model = []
    # 导入参数配置
    import_statements = ["from .%s import %s" % (arch, model_name)]
    pytorch_modules = None
    if len(subgraphs) > 1:
        # 把子图融合成一个总体module
        python_modules, pytorch_modules, subgraph_inputs, subgraph_outputs = \
            fuse_subgraphs_to_module(full_graph, subgraphs, model_name,
                                     initialize_weights, model_template_filename,
                                     os.path.join(output_directory, "%s.py" % arch))

