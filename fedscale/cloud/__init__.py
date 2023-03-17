def get_group_name(args):
    partitioner = "trace" if args.data_map_file else args.partition_method
    group = "trace-" if args.data_map_file else "n%d-" % args.total_participants
    group += "s%d-%s-" % (args.num_participants, partitioner)
    if partitioner == "dirichlet":
        group += "%f-" % args.dirichlet_alpha
    group += "r%d-%s-%s-p%d-%s" % (args.rounds, args.sample_mode, args.model, args.parallel_sessions, args.time_stamp)
    return group
