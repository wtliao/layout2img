import matplotlib.pyplot as plt


def show_results(experiment):
    results = experiment['results']
    labels = experiment['labels']

    for i, (dataset, d_value) in enumerate(results.items()):
        f = plt.figure(figsize=(16., 10.))
        plt.suptitle(dataset, fontsize=20, fontweight='bold')
        girds = {6: (2, 3), 12: (3, 4)}[len(results[dataset])]
        for idx, (n, n_value) in enumerate(results[dataset].items()):
            plt.subplot(girds[0], girds[1], idx+1)
            plt.pie(n_value, explode=(0.05, 0), labels=labels, 
                    colors=['coral', 'skyblue'], autopct='%.1f%%', shadow=False, 
                    startangle=90, textprops={'fontsize':12, 'color': 'k'})
            plt.title(n, fontsize=14, fontweight='bold')
    plt.show()

def show_results_inbar(experiment):
    results = experiment['results']
    labels = list(experiments[0]['results']['coco'].keys())+['mean']
    size = 11

    x = np.arange(size)
    r1 = np.array(list(experiments[0]['results']['coco'].values()))
    r2 = np.array(list(experiments[0]['results']['vg'].values()))
    r3 = np.array(list(experiments[1]['results']['coco'].values()))
    r4 = np.array(list(experiments[1]['results']['vg'].values()))
    
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, r1, width=width, label='Ours-D on COCO')
    plt.bar(x + width, r2, width=width, label='Ours-D on VG')
    plt.bar(x + 2 * width, r3, width=width, label='Ours-ED on COCO')
    plt.bar(x + 3 * width, r4, width=width, label='Ours-ED on VG')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # lostgan_results = {
    #     'coco': {
    #         'P0': [48, 52], 'P1': [65, 35], 'P2': [50, 50], 'P3': [49, 51], 'P4': [60, 40], 'P5': [57, 43], 
    #         'P6': [50, 50], 'P7': [58, 42], 'P8': [62, 38], 'P9': [48, 52], 'P10': [49, 51], 'all': [596, 504]
    #     }, 
    #     'vg': {
    #         'P0': [50, 50], 'P1': [53, 47], 'P2': [53, 47], 'P3': [61, 39], 'P4': [45, 55], 'P5': [62, 38], 
    #         'P6': [61, 39], 'P7': [63, 37], 'P8': [55, 45], 'P9': [53, 47], 'P10': [56, 44], 'all': [612, 488]
    #     }
    # }

    # layout2im_results = {
    #     'coco': {
    #         'P0': [49, 51], 'P1': [56, 44], 'P2': [46, 54], 'P3': [40, 60], 'P4': [52, 48], 'all': [243, 257]
    #     }, 
    #     'vg': {
    #         'P0': [40, 60], 'P1': [49, 51], 'P2': [50, 50], 'P3': [50, 50], 'P4': [45, 55], 'all': [234, 266]
    #     }
    # }

    lostgan_results = {
        'coco': {
            'P1': 57, 'P2': 50, 'P3': 49, 'P4': 60, 'P5': 57, 'P6': 58, 'P7': 62, 'P8': 48, 'P9': 49, 'P10': 48, 'all': 54.3
        }, 
        'vg': {
            'P1': 53, 'P2': 53, 'P3': 61, 'P4': 45, 'P5': 62, 'P6': 61, 'P7': 63, 'P8': 55, 'P9': 53, 'P10': 56, 'all': 56.7
        }
    }

    layout2im_results = {
        'coco': {
            'P1': 56, 'P2': 48, 'P3': 47, 'P4': 52, 'P5': 49, 'P6': 56, 'P7': 48, 'P8': 47, 'P9': 52, 'P10': 49, 'all': 52.3
        }, 
        'vg': {
            'P1': 53, 'P2': 48, 'P3': 49, 'P4': 56, 'P5': 54, 'P6': 55, 'P7': 53, 'P8': 55, 'P9': 54, 'P10': 51, 'all': 52.8
        }
    }

    experiments = [
        {'results': lostgan_results, 'labels': ['ours-D', 'lostgan']}, 
        {'results': layout2im_results, 'labels': ['ours-ED', 'layout2im']}
    ]

    # show_results(experiments[0])
    show_results_inbar(experiments)
