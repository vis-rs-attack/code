base_config_v1 = {
    'steps': 20,
    'epsilon': 1/255,
    'gamma': 1,
    'seed': 0,
    'user': 9,
}

grid_config_v1 = {  
    'blackbox': [0, 1],
    'from_rank': [2000, 20000],
    'examples': [32, 128, 2048],
    'do_pca': [False, True],
    'n_components': [200, 350, 500],
    'by_rank': [False, True],
    'rank_distribution': ['normal', 'uniform'],
}

alreay_perfromed = {

}

def gen_conf(grid):
    for blackbox in grid['blackbox']:
        if not blackbox:
            for rank in grid['from_rank']:
                yield f"wb_f{rank}", {
                    'blackbox': blackbox,
                    'from_rank': rank,
                }
        else:
            for examples in grid['examples']:
                for rank in grid['from_rank']:
                    for do_pca in grid['do_pca']:
                        if not do_pca:
                            for by_rank in grid['by_rank']:
                                if not by_rank:
                                    yield f"bb_f{rank}_ex{examples}", {
                                        'blackbox': blackbox,
                                        'from_rank': rank,
                                        'do_pca': do_pca,
                                        'by_rank': by_rank,
                                        'examples': examples,
                                    }
                                else:
                                    for rank_distribution in grid['rank_distribution']:
                                        yield f"bb_f{rank}_ex{examples}_{rank_distribution[0]}Rank", {
                                            'blackbox': blackbox,
                                            'from_rank': rank,
                                            'do_pca': do_pca,
                                            'by_rank': by_rank,
                                            'rank_distribution': rank_distribution,
                                            'examples': examples,
                                    }
                        else:
                            for n_components in grid['n_components']:
                                for by_rank in grid['by_rank']:
                                    if not by_rank:
                                        yield f"bb_f{rank}_ex{examples}_pca{n_components}", {
                                            'blackbox': blackbox,
                                            'from_rank': rank,
                                            'do_pca': do_pca,
                                            'n_components': n_components,
                                            'by_rank': by_rank,
                                            'examples': examples,
                                        }
                                    else:
                                        for rank_distribution in grid['rank_distribution']:
                                            yield f"bb_f{rank}_ex{examples}_pca{n_components}_{rank_distribution[0]}Rank", {
                                                'blackbox': blackbox,
                                                'from_rank': rank,
                                                'do_pca': do_pca,
                                                'n_components': n_components,
                                                'by_rank': by_rank,
                                                'rank_distribution': rank_distribution,
                                                'examples': examples,
                                        }



