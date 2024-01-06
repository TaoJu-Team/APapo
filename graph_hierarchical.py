# -*- coding:utf-8 -*-

import math
import os

import graph
from collections import OrderedDict
import time


def compute_partitioning(compute_times, activation_sizes, parameter_sizes,
                         output_activation_sizes, all_predecessor_ids,
                         num_machines, network_bandwidth):
    A = []  # (101, 100, 4, 3)
    for i in range(len(compute_times)):
        row_A = []
        for j in range(len(compute_times[0])):
            row_row_A = []
            for m in range(num_machines):
                row_row_A.append((None, None, None))
            row_A.append(row_row_A)
        A.append(row_A)

    for i in range(len(compute_times)):
        for j in range(i, len(compute_times[0])):
            cum_compute_time = compute_times[i][j]  # i --> j 的计算时间
            cum_activation_size = activation_sizes[i][j]  # i --> j 的激活大小
            cum_parameter_size = parameter_sizes[i][j]  # i --> j 的参数大小
            max_m = 1 if straight_pipeline else num_machines
            for m in range(max_m):  # 遍历流水线下一阶段的机器,0,1,2,3
                stashed_data_size = math.ceil((num_machines - (m + 1)) / (m + 1)) * (
                        cum_activation_size + cum_parameter_size)
                if use_memory_constraint and stashed_data_size > memory_size:
                    continue

                # communication_time = (2 * cum_parameter_size / network_bandwidth)

                if cum_compute_time is None:
                    A[i][j][m] = (None, None, None)
                else:
                    A[i][j][m] = (sum([cum_compute_time]), None, m + 1)

    min_machines = 1
    max_i = 1
    for i in range(max_i):  # 1
        for m in range(min_machines, num_machines):  # 遍历下一阶段机器的可能选择 1,2,3
            for j in range(i + 1, len(compute_times[0])):  # 100
                (min_pipeline_time, optimal_split, optimal_num_machines) = A[i][j][m]  # 2,3,4
                for k in all_predecessor_ids[j]:  # 前序state
                    if i > 0 and k in all_predecessor_ids[i - 1]:
                        continue

                    max_m_prime = m + 1  # 2,3,4
                    for m_prime in range(1, max_m_prime):
                        # 输入传输时间 input_transfer_time 使用 k 的输出激活尺寸计算
                        # 2.0 : 激活和梯度
                        # 将需要传输的数据量除以通信链路的网络带宽来估计通信时间
                        input_transfer_time = (2.0 * output_activation_sizes[k]) / (network_bandwidth)
                        output_transfer_time = None
                        if j < len(output_activation_sizes) - 1:  # j<99
                            # 输出传输时间 output_transfer_time 使用 j 的输出激活尺寸计算
                            output_transfer_time = (2.0 * output_activation_sizes[j]) / (network_bandwidth)
                        last_stage_time = compute_times[k + 1][j]
                        if last_stage_time is None:
                            continue
                        last_stage_parameter_size = parameter_sizes[k + 1][j]
                        stashed_data_size = activation_sizes[k + 1][j] + last_stage_parameter_size
                        # stashed_data_size *= math.ceil((num_machines - (m + 1)) / m_prime)
                        if use_fewer_machines and stashed_data_size > memory_size:
                            continue
                        # last_stage_time 是 (k 到 j 的计算时间) + 传输时间
                        last_stage_time = sum([last_stage_time, (2 * last_stage_parameter_size) / network_bandwidth])

                        if A[i][k][m - m_prime][0] is None:
                            continue

                        pipeline_time = max(A[i][k][m - m_prime][0], last_stage_time)
                        if activation_compression_ratio is not None:
                            input_transfer_time /= activation_compression_ratio
                            if output_transfer_time is not None:
                                output_transfer_time /= activation_compression_ratio
                            pipeline_time = max(pipeline_time, input_transfer_time)
                            if output_transfer_time is not None:
                                pipeline_time = max(pipeline_time, output_transfer_time)
                        if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                            optimal_split = (k, m - m_prime)  # 选一个优化分割点，k为分割点，使用m-m_prime个机器
                            optimal_num_machines = m_prime  # 划分出的数据并行阶段使用多少个机器
                            min_pipeline_time = pipeline_time
                A[i][j][m] = (min_pipeline_time, optimal_split, optimal_num_machines)
    return A


# 分析阶段
def analyze_partitioning(A, states, start, end, network_bandwidth, num_machines,
                         activation_compression_ratio, print_configuration, verbose=False):
    metadata = A[start][end - 1][num_machines - 1]  # 最初的states(0~99),4台机器
    next_split = metadata[1]  # metadata[1] 是 optimal_split，即 (k, m-m_prime)
    remaining_machines_left = num_machines
    splits = []
    stage_num_machines = []
    prev_split = end - 1  # 除去末尾state，最终加上
    while next_split is not None:  # 是否继续分割的条件
        num_machines_used = metadata[2]
        if verbose:
            print("-------------------------------------")
            print("Number of machines used: %d..." % num_machines_used)
            print("Split between layers %d and %d..." % (next_split[0], next_split[0] + 1))
            print("Split before antichain %s..." % (states[next_split[0] + 1].antichain))
        splits.append(next_split[0] + 1)
        compute_time = states[prev_split - 1].compute_time - states[next_split[0]].compute_time
        pp_communication_time_input = (  # 下个阶段的数据输入时间
                                              2.0 * states[next_split[0]].output_activation_size *
                                              (1.0 / float(num_machines_used))) / network_bandwidth
        pp_communication_time_output = (  # 上个阶段的数据输出时间
                                               2.0 * states[prev_split - 1].output_activation_size *
                                               (1.0 / float(num_machines_used))) / network_bandwidth
        if activation_compression_ratio is not None:
            pp_communication_time_input /= activation_compression_ratio
            pp_communication_time_output /= activation_compression_ratio
        if activation_compression_ratio is None:
            pp_communication_time_input = 0.0
            pp_communication_time_output = 0.0

        compute_time /= num_machines_used  # 本阶段运行时间

        if verbose:
            print(("Compute time = %f, Pipeline-parallel communication time = %f...") %
                  (compute_time, max(pp_communication_time_input, pp_communication_time_output)))
        prev_split = splits[-1]  # 设定新的末尾边界
        metadata = A[start][next_split[0]][next_split[1]]  # next_split = metadata[1]
        next_split = metadata[1]  # 设定新的下一次分割点
        stage_num_machines.append(num_machines_used)  # 每个stage使用的机器数
        remaining_machines_left -= num_machines_used  # 剩余机器数

    # 最终剩下的stage
    if verbose:
        print("-------------------------------------")
        print("Number of machines used: %d..." % metadata[2])
    num_machines_used = metadata[2]
    remaining_machines_left -= num_machines_used
    compute_time = states[prev_split - 1].compute_time
    compute_time /= num_machines_used
    if verbose:
        print("Compute time = %f..." % compute_time)
        print("-------------------------------------")

    if print_configuration:
        print("Number of machines in budget not used: %d..." %
              remaining_machines_left)
        print()
        print("(Split start, split end) / compute time taken per stage / num machines per stage")

    prev_split = start
    splits.reverse()
    splits.append(end)
    stage_num_machines.append(num_machines_used)
    stage_num_machines.reverse()
    for i in range(len(splits)):
        time = 0.0
        if prev_split > 0:
            time = states[splits[i] - 1].compute_time - states[prev_split - 1].compute_time
        else:
            time = states[splits[i] - 1].compute_time
        if print_configuration:
            print((prev_split, splits[i]), '\t\t', time, '\t\t', stage_num_machines[i])
        prev_split = splits[i]
    if print_configuration:
        print()
    return splits[:-1]  # 最后一个不返回


def main(all_num_machines, profile_filename, network_bandwidths, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression_ratio, output_directory,
         print_configuration=True, verbose=False):
    gr = graph.Gragh.from_str(open(profile_filename, 'r').read())  # 加载profile文件构建计算图

    # 对图的输入进行处理
    sources = gr.sources()
    nodes_to_remove = OrderedDict()  # 会记住元素的插入顺序
    for source in sources:
        if source.node_desc.startswith("Input"):
            source.forward_compute_time = 0.0
            source.backward_compute_time = 0.0
            source.activation_size = 0.0
            source.parameter_size = 0.0
            nodes_to_remove[source] = []
            for out_node in gr.edges[source.node_id]:
                nodes_to_remove[source].append(out_node)  # 记录source节点的出边节点
            gr.remove_node(source)  # 在图中移除input source

    # 对图的输出进行处理
    sinks = gr.sinks()  # 移除没有用到的输出
    for sink in sinks:
        if sink.node_desc.startswith("__getitem__"):  # 这里针对transformer中的getitem
            gr.remove_node(sink)

    # print(gr)
    antichain_gr = gr.antichain_dag()  # 得到反链DAG，gr不包含sources,sinks
    # print(antichain_gr)
    # print(antichain_gr) # resNet18节点71个，不一定DAG为71个，有很多出边节点相同的边，也是链
    states = antichain_gr.topological_sort()  # DAG拓扑排序


    if verbose:  # resnet18的antichain生成100个
        print("Total number of states: %d" % len(states))

    #####################################
    # 计算分区准备阶段
    #####################################
    states_indices = {}
    for i in range(len(states)):
        print(states[i])
        states_indices[states[i]] = i  # 为DAG每个节点（反链）编号

    # 记录每个state的激活值大小
    for i in range(len(states)):
        for antichain_node in states[i].antichain:
            # 计算每个DAG节点激活值大小
            states[i].output_activation_size += gr.nodes[antichain_node].activation_size

    # 记录每个state的前序节点给自己的输出叠加
    for i in range(len(states)):
        antichain = states[i].antichain
        all_predecessors = gr.all_predecessors(antichain)
        states[i].compute_time = 0.0
        states[i].activation_size = 0.0
        states[i].parameter_size = 0.0
        for predecessor in all_predecessors:
            states[i].compute_time += ((predecessor.forward_compute_time +
                                        predecessor.backward_compute_time) / 1000)  # /1000是s
            # 输出激活值大小,把其增强反链的前序节点给自己的输出都叠加起来
            states[i].activation_size += predecessor.activation_size
            states[i].parameter_size += predecessor.parameter_size
    gr.reset()  # 前序、后序节点的字典设为空，后续再使用

    output_activation_sizes = [state.output_activation_size for state in states]
    # 记录每个state前置state的id
    all_predecessor_ids = [[states_indices[predecessor] for predecessor in
                            antichain_gr.predecessors(states[i].node_id)]
                           for i in range(len(states))]

    # 动态规划需要保存的数据，每个state到后续state的C，A，P
    compute_times = []
    activation_sizes = []
    parameter_sizes = []
    for i in range(len(states) + 1):
        compute_times_row = []
        activation_sizes_row = []
        parameter_sizes_row = []
        for j in range(len(states)):
            if i == 0:
                compute_times_row.append(states[j].compute_time)
                activation_sizes_row.append(states[j].activation_size)
                parameter_sizes_row.append(states[j].parameter_size)
            else:
                if j > (i - 1):
                    compute_times_row.append(states[j].compute_time -
                                             states[i - 1].compute_time)
                    activation_sizes_row.append(states[j].activation_size -
                                                states[i - 1].activation_size)
                    parameter_sizes_row.append(states[j].parameter_size -
                                               states[i - 1].parameter_size)
                else:
                    compute_times_row.append(None)
                    activation_sizes_row.append(None)
                    parameter_sizes_row.append(None)
        compute_times.append(compute_times_row)
        activation_sizes.append(activation_sizes_row)
        parameter_sizes.append(parameter_sizes_row)

    #####################################
    # 计算阶段
    #####################################
    all_As = []
    for num_machines, network_bandwidth in zip(all_num_machines, network_bandwidths):
        print("Solving optimization problem with %d machines with intra-machine bandwidth of %.2f GB/s"
              % (num_machines, network_bandwidth / 10 ** 9))
        A = compute_partitioning(compute_times, activation_sizes, parameter_sizes,
                                 output_activation_sizes, all_predecessor_ids,
                                 num_machines, network_bandwidth)
        all_As.append(A)
    #####################################
    # 分析阶段
    #####################################
    splits = [(0, len(states))]
    print("======================================")
    print("\t\t analysis result")
    print("======================================")
    new_splits = []
    stage_id = 0
    for (start, end) in splits:
        partial_splits = analyze_partitioning(all_As[0], states, start, end,
                                              network_bandwidths[0], all_num_machines[0],
                                              activation_compression_ratio,
                                              print_configuration, verbose)
        start_point = start
        print(partial_splits)
        for split in partial_splits:
            new_splits.append((start_point, split))
            predecessors = gr.all_predecessors(states[split - 1].antichain)
            for predecessor in predecessors:
                if predecessor.stage_id is None:
                    predecessor.set_stage_id(stage_id)
            start_point = split
            stage_id += 1
        new_splits.append((start_point, end))

        predecessors = gr.all_predecessors(states[end - 1].antichain)
        for predecessor in predecessors:
            if predecessor.stage_id is None:
                predecessor.set_stage_id(stage_id)
        stage_id += 1
    print("Total number of stages: %d" % stage_id)

    # 以下是为了把图写到文件之中
    for source in nodes_to_remove:  # 之前移除了input节点，现在需要加回到图中
        for out_node in nodes_to_remove[source]:  # input对应的哪些输出
            source.stage_id = 0
            gr.add_edge(source, out_node)

    # 将切分好的图写到文件里
    if output_directory is not None:
        total_num_machines = 1
        for num_machines in all_num_machines:
            total_num_machines *= num_machines
        gr.to_dot(os.path.join(output_directory, "gpu=%d" % total_num_machines))
        gr_str = str(gr)
        with open(os.path.join(output_directory, "gpu=%d.txt" % total_num_machines), 'w') as f:
            f.write(gr_str)

    # 以下是为了做分析对比
    # 用动态规划算法得出来的优化时间
    pipeline_parallel_total_time = A[0][len(states) - 1][num_machines - 1][0]
    if verbose:
        print()
        print("Time per stage in pipeline:", pipeline_parallel_total_time)


if __name__ == '__main__':
    all_num_machines = [4]
    profile_filename = 'graph/graph_resnet50.txt'
    network_bandwidths = [100 * 10 ** 9]
    assert (len(all_num_machines) == len(network_bandwidths))
    memory_size = 32000000000
    straight_pipeline = False  # 不适用单模型并行
    output_directory = "../DP_Partition/Partition_result/resnet50__8"
    use_memory_constraint = False
    use_fewer_machines = False
    activation_compression_ratio = None

    times = time.time()
    main(all_num_machines, profile_filename, network_bandwidths, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression_ratio, output_directory,
         verbose=True)
    print(time.time() - times)
