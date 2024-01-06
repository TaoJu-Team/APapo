# -*- coding:utf-8 -*-

import graphviz

class Gragh(object):
    def __init__(self, node=None):
        self.nodes = {}  # 节点
        if node is not None:
            self.nodes[node.node_id] = node
        self.edges = {}  # 出边
        self.in_edges = {}  # 入边
        '''以单个下划线开头的变量或方法仅供内部使用'''
        self._antichain_dag = None  # 反链DAG
        self._augmented_antichains = {}  # 增强反链
        self._next_antichains = {}  # 后续反链
        self._predecessors = {}  # 节点的前序节点
        self._successors = {}  # 节点的后续节点
        self._deaugment_augmented_antichains = {}

        self._colors = ['lightblue', 'green', 'grey', 'firebrick1',
                        'gold', 'chocolate1', 'beige']

    # 获取input节点
    def sources(self):
        sources = []
        for node_id in self.nodes:
            # 节点不在入边字典中或节点的入边字典长度为0
            # 即input节点
            if node_id not in self.in_edges or len(self.in_edges[node_id]) == 0:
                sources.append(self.nodes[node_id])
        return sources

    # 移除节点
    def remove_node(self, node):
        del self.nodes[node.node_id]  # 删除引用，释放内存
        if node.node_id in self.edges:
            out_nodes = self.edges[node.node_id]
            del self.edges[node.node_id]
            for out_node in out_nodes:
                self.in_edges[out_node.node_id].remove(node)
        if node.node_id in self.in_edges:
            in_nodes = self.in_edges[node.node_id]
            del self.in_edges[node.node_id]
            for in_node in in_nodes:
                self.edges[in_node.node_id].remove(node)

    # 获取没有用到的输出
    def sinks(self):
        sinks = []
        for node_id in self.nodes:
            # 节点不在出边字典中或节点的出边字典长度为0
            # 即没有用到的输出
            if node_id not in self.edges or len(self.edges[node_id]) == 0:
                sinks.append(self.nodes[node_id])
        return sinks

    # 获取节点的前序节点
    def predecessors(self, node):
        if node in self._predecessors:
            return self._predecessors[node]
        predecessors = set()
        if node not in self.in_edges:  # source节点
            return predecessors  # 返回空
        for in_node in self.in_edges[node]:
            predecessors.add(in_node)
            '''update:将集合a和集合b取并集，并将结果保存在a中，对象b不改变，但是没有返回值'''
            predecessors.update(self.predecessors(in_node.node_id))  # 递归遍历所有前序节点
        self._predecessors[node] = predecessors
        return self._predecessors[node]

    # 获取增强反链中节点的所有前序节点
    def all_predecessors(self, antichain):
        all_predecessors = set()
        for antichain_node in antichain:
            all_predecessors.update(self.predecessors(antichain_node))
            all_predecessors.add(self.nodes[antichain_node])
        return all_predecessors

    # 获取节点的所有后续节点
    def successors(self, node):
        if node in self._successors:
            return self._successors[node]
        successors = set()
        if node not in self.edges:  # sink节点
            return successors
        for out_node in self.edges[node]:
            successors.add(out_node)
            successors.update(self.successors(out_node.node_id))
        self._successors[node] = successors
        return self._successors[node]

    # 反链节点计算图中插入边，并非最初的节点之间
    def add_edge(self, node1, node2):
        if node1.node_id not in self.nodes:
            self.nodes[node1.node_id] = node1
        if node2.node_id not in self.nodes:
            self.nodes[node2.node_id] = node2

        if node2.node_id not in self.in_edges:
            self.in_edges[node2.node_id] = list()
        self.in_edges[node2.node_id].append(node1)
        if node1.node_id not in self.edges:
            self.edges[node1.node_id] = list()
        self.edges[node1.node_id].append(node2)

    def reset(self):
        self._predecessors = {}
        self._successors = {}

    # 使用深度优先遍历（DFS）的拓扑排序
    def topological_sort_helper(self, node_id):
        if node_id in self.marked_nodes:
            return
        if node_id in self.temporarily_marked_nodes:
            raise Exception("Graph has a cycle")
        self.temporarily_marked_nodes.add(node_id)
        if node_id in self.edges:
            out_nodes = list(self.edges[node_id])
            out_nodes.sort(key=lambda x: (x.node_desc, x.height))
            for out_node in out_nodes:
                self.topological_sort_helper(out_node.node_id)
        self.marked_nodes.add(node_id)
        self.temporarily_marked_nodes.remove(node_id)
        self.sorted_nodes.insert(0, node_id)

    # 使用深度优先遍历（DFS）的拓扑排序
    def topological_sort(self):
        # Algorithm from https://en.wikipedia.org/wiki/Topological_sorting
        self.sorted_nodes = []
        self.marked_nodes = set()
        self.temporarily_marked_nodes = set()
        nodes = list(self.nodes.values())
        nodes.sort(key=lambda x: x.node_desc)
        for node in nodes:
            if node.node_id in self.marked_nodes:
                continue
            self.topological_sort_helper(node.node_id)
        return [self.nodes[node_id] for node_id in self.sorted_nodes]

    # 增强反链，对每个节点，找到其增强反链
    def augment_antichain(self, antichain):
        antichain_key = tuple(sorted(antichain))
        if antichain_key in self._augmented_antichains:
            return self._augmented_antichains[antichain_key]
        extra_nodes = set()
        # all_predecessors = set()
        # for antichain_node in antichain:
        #     # 获取所有的前序节点
        #     predecessors = self.predecessors(antichain_node)
        #     # 返回两个集合的并集，并将并集作为一个新的对象的返回
        #     # 即包含了所有集合的元素，重复的元素只会出现一次
        #     all_predecessors = all_predecessors.union(predecessors)
        for antichain_node in antichain:
            predecessors = self.predecessors(antichain_node)
            for predecessor in predecessors:
                # 对每个前序节点的出边
                for out_node in self.edges[predecessor.node_id]:
                    # 如果出边不在前序节点列表之中，且 出边节点不是本节点
                    if out_node not in predecessors and out_node.node_id != antichain_node:
                        extra_nodes.add(predecessor.node_id)
        # 节点的增强反链是部分前序节点+自身
        self._augmented_antichains[antichain_key] = list(extra_nodes) + antichain
        # print(' %s的增强反链：'%antichain_key[0],list(extra_nodes) + antichain)
        return self._augmented_antichains[antichain_key]

    # 用来判断某新节点是否为后续反链
    def is_next_antichain(self, augmented_antichain, new_node):
        successors = self.successors(new_node)
        augmented_antichain_set = set(augmented_antichain)
        for successor in successors:
            # 如果后续节点有一个在增强节点之中，就返回false，说明不是后续反链
            if successor.node_id in augmented_antichain_set:
                return False
        return True

    # 构建图分割点之间的依赖关系
    def deaugment_augmented_antichain(self, augmented_antichain):
        augmented_antichain_key = tuple(sorted(augmented_antichain))
        if augmented_antichain_key in self._deaugment_augmented_antichains:
            return self._deaugment_augmented_antichains[augmented_antichain_key]
        nodes_to_remove = set()
        for augmented_antichain_node in augmented_antichain:
            successors = self.successors(augmented_antichain_node)
            for augmented_antichain_node_prime in augmented_antichain:
                if self.nodes[augmented_antichain_node_prime] in successors:
                    nodes_to_remove.add(augmented_antichain_node)
        antichain = list()
        for augmented_antichain_node in augmented_antichain:
            if (augmented_antichain_node not in nodes_to_remove and augmented_antichain_node not in antichain):
                antichain.append(augmented_antichain_node)
        self._deaugment_augmented_antichains[augmented_antichain_key] = antichain
        # print(augmented_antichain_key, antichain)
        return self._deaugment_augmented_antichains[augmented_antichain_key]

    # 构建反链关系，保存可分割反链之间依赖
    def construct_antichain(self, augmented_antichain, old_node, new_node):
        new_antichain = [x if x != old_node else new_node for x in augmented_antichain]
        return self.deaugment_augmented_antichain(new_antichain)

    # 根据source节点获取后续反链
    # 在添加后续的节点来构建整个计算图
    def next_antichains(self, antichain):
        antichain_key = tuple(sorted(antichain))
        if antichain_key in self._next_antichains:
            return self._next_antichains[antichain_key]

        next_antichains = []
        antichain_set = set(antichain)
        augmented_antichain = self.augment_antichain(antichain)
        for augmented_antichain_node in augmented_antichain:
            next_nodes = self.edges[augmented_antichain_node] if augmented_antichain_node in self.edges else []
            for next_node in next_nodes:
                if next_node.node_id in antichain_set:  # 如果出边节点已经在反链集合之中，跳过，进入下一循环
                    continue
                # 判断每个节点增强反链中节点的出边节点，是否为此增强反链的后续反链
                if self.is_next_antichain(augmented_antichain, next_node.node_id):
                    # 然后使用本增强反链、增强反链和可以作为其后继反链的节点
                    # 构建新的反链，称为图的分割点
                    next_antichain = self.construct_antichain(augmented_antichain,
                                                              augmented_antichain_node,
                                                              next_node.node_id)
                    next_antichains.append(next_antichain)
        self._next_antichains[antichain_key] = next_antichains  # 保存每个增强反链的后续反链
        return self._next_antichains[antichain_key]  # 返回每个增强节点的后续反链

    # 反链DAG
    def antichain_dag(self):
        if self._antichain_dag is not None:
            return self._antichain_dag

        antichain_dag = Gragh()
        antichain_id = 0
        antichain = [self.sources()[0].node_id]  # 除去source,sink节点，即下一个节点作为初始节点
        source_node = AntichainNode("antichain_%d" % antichain_id,
                                    self.augment_antichain(antichain))
        antichain_dag.sources = source_node
        antichain_queue = [antichain]
        antichain_mapping = {tuple(sorted(antichain)): source_node}  # 构建DAG的反链依赖信息

        while len(antichain_queue) > 0:
            antichain = antichain_queue.pop(0)  # 得到队列第一个元素
            antichain_key = tuple(sorted(antichain))
            if antichain_key in self._next_antichains:
                continue
            next_antichains = self.next_antichains(antichain)  # 依据source得到后续反链

            for next_antichain in next_antichains:
                next_antichain_key = tuple(sorted(next_antichain))
                if next_antichain_key not in antichain_mapping:
                    antichain_id += 1
                    next_antichain_node = AntichainNode("antichain_%d" % antichain_id,
                                                        self.augment_antichain(next_antichain))
                    antichain_mapping[next_antichain_key] = next_antichain_node
                # 向 反链DAG 插入边：其中节点信息，出边入边信息都被放入
                # 这个边的依赖和之前graph之间不同，这个是各个反链之间的边依赖
                antichain_dag.add_edge(antichain_mapping[antichain_key],
                                       antichain_mapping[next_antichain_key])
                antichain_queue.append(next_antichain)  # 循环继续，来构建整个DAG
        self._antichain_dag = antichain_dag
        return antichain_dag

    # 模型切分后使用不同颜色表示各个stage
    def to_dot(self, arch):
        dot = graphviz.Digraph()
        for node in self.nodes.values():
            node_desc = "%s\n[forward_compute_time=%.3f,backward_compute_time=%.3f,activation_size=%s,parameter_size=%.1f]" % (
                node.node_desc, node.forward_compute_time, node.backward_compute_time,
                node.activation_size, node.parameter_size)
            if node.stage_id is not None:
                color = self._colors[node.stage_id % len(self._colors)]
                dot.node(node.node_id, node_desc,
                         color=color, style='filled')
            else:
                dot.node(node.node_id, node_desc)
        for node in self.nodes.values():
            if node.node_id not in self.edges:
                continue
            for out_node in self.edges[node.node_id]:
                dot.edge(node.node_id, out_node.node_id)
        dot.render(arch)

    def __str__(self):
        strs = []
        for node in self.nodes.values():
            strs.append(str(node))
        for node in self.nodes.values():
            if node.node_id not in self.in_edges:
                continue
            for in_node in self.in_edges[node.node_id]:
                strs.append("\t%s -- %s" % (in_node.node_id, node.node_id))
        return "\n".join(strs)

    def add_node(self, node):
        self.nodes[node.node_id] = node

    # 若无分区，则复制整个计算图
    def copy(self):
        gr = Gragh()
        for node_id in self.in_edges:
            for node2 in self.in_edges[node_id]:
                gr.add_edge(node2, self.nodes[node_id])
        return gr

    # 按照stage_id构建子图
    def partition_graph(self):
        stage_ids = set()
        for node_id in self.nodes:
            stage_ids.add(self.nodes[node_id].stage_id)
        if len(stage_ids) == 1:
            return self.copy()
        subgraphs = []
        for stage_id in stage_ids:
            subgraphs.append(self.partition_graph_helper(stage_id))
        return subgraphs

    # 针对给定的stage，在所有节点中查找对应stage的节点，构建一个子图
    def partition_graph_helper(self, stage_id):
        subgraph = Gragh()
        for node1_id in self.nodes:
            if self.nodes[node1_id].stage_id == stage_id:
                subgraph.add_node(self.nodes[node1_id])
                if node1_id not in self.edges: continue
                for node2 in self.edges[node1_id]:
                    if node2.stage_id == stage_id:
                        subgraph.add_edge(self.nodes[node1_id], node2)
        return subgraph

    # Helper方法，它用距sink的深度注释图中的每个节点
    def populate_depths(self):
        sources = self.sources()
        sources[0].depth = 1
        queue = [sources[0]]
        while len(queue) > 0:
            node = queue.pop(-1)
            if node.node_id not in self.edges: continue
            for out_node in self.edges[node.node_id]:
                if out_node.depth is None or out_node.depth < (node.depth + 1):
                    out_node.depth = node.depth + 1
                queue.append(out_node)

    @staticmethod
    def from_str(graph_str):
        gr = Gragh()
        graph_str_lines = graph_str.strip().split('\n')  # 删除字符串首尾的空格，以'\n'为分隔符切片，返回字符串列表
        for graph_str_line in graph_str_lines:
            if not graph_str_line.startswith('\t'):  # 构建节点
                node = Node.from_str(graph_str_line.strip())
                gr.nodes[node.node_id] = node  # 添加到计算图中
            else:  # 构建边
                # eg: node10 --> node11
                [in_node_id, node_id] = graph_str_line.strip().split(" -- ")
                # 入边
                if node_id not in gr.in_edges:  # 入边节点不在入边字典中
                    gr.in_edges[node_id] = [gr.nodes[in_node_id]]
                else:
                    gr.in_edges[node_id].append(gr.nodes[in_node_id])
                # 出边
                if in_node_id not in gr.edges:  # 出边节点不在出边字典中
                    gr.edges[in_node_id] = [gr.nodes[node_id]]
                else:
                    gr.edges[in_node_id].append(gr.nodes[node_id])
        return gr


class Node(object):
    def __init__(self, node_id, node_desc="",
                 forward_compute_time=0.0, backward_compute_time=0.0,
                 activation_size=0.0, parameter_size=0.0,
                 stage_id=None):
        self.node_id = node_id
        self.node_desc = node_desc
        self.forward_compute_time = forward_compute_time
        self.backward_compute_time = backward_compute_time
        self.activation_size = activation_size  # 激活值大小
        self.parameter_size = parameter_size
        self.stage_id = stage_id  # 分区编号
        self.depth = None  # 距sink的深度
        self.height = None  # 拓扑排序需要用到

    def set_stage_id(self, stage_id):
        self.stage_id = stage_id

    def __str__(self):
        stage_id_str = " -- stage_id=%d" % self.stage_id if self.stage_id is not None else ""
        node_desc = self.node_desc.replace('\n', "")
        activation_size = ("%s" % self.activation_size).replace(", ", "; ")
        return "%s -- %s -- forward_compute_time=%.3f, backward_compute_time=%.3f, activation_size=%s, parameter_size=%.3f%s" % (
            self.node_id, node_desc, self.forward_compute_time, self.backward_compute_time,
            activation_size, self.parameter_size, stage_id_str)

    @staticmethod
    def from_str(node_str):  # 构建节点
        # eg: node1 -- Input0 -- forward_compute_time=0.000, backward_compute_time=0.000, activation_size=0.0, parameter_size=0.000
        node_str_tokens = node_str.strip().split(" -- ")
        node_id = node_str_tokens[0]
        node_desc = node_str_tokens[1]
        node_metadata = node_str_tokens[2]
        stage_id = None

        if len(node_str_tokens) > 3:
            stage_id = int(node_str_tokens[3].split("=")[1])  # 分区编号
        [forward_compute_time, backward_compute_time, activation_size, parameter_size] = node_metadata.split(", ")
        forward_compute_time = float(forward_compute_time.split("=")[1])
        backward_compute_time = float(backward_compute_time.split("=")[1])
        if "[" in activation_size:
            activation_size = activation_size.split("=")[1]
            activation_size = sum([float(x) for x in activation_size.lstrip("[").rstrip("]").split("; ")])
        else:
            activation_size = float(activation_size.split("=")[1])
        parameter_size = float(parameter_size.split("=")[1])

        return Node(node_id, node_desc, forward_compute_time=forward_compute_time,
                    backward_compute_time=backward_compute_time, activation_size=activation_size,
                    parameter_size=parameter_size, stage_id=stage_id)


class AntichainNode(Node):
    def __init__(self, node_id, antichain, node_desc=""):
        self.antichain = antichain  # 每个分割点所包含的节点
        self.output_activation_size = 0.0  # 每个分割点包含的节点的激活值之和
        super(AntichainNode, self).__init__(node_id, node_desc)

    def __str__(self):
        return "%s -- %s" % (self.node_id, self.antichain)
