
import sklearn.datasets as datasets
from matplotlib import pyplot as plt
import numpy as np
import math

iris = datasets.load_iris()
# label_dict={'blue':"Setosa",'red':"Versicolour",'green':"Virginica"}
label_dict = {0: 'Setosa', 1: 'Versicolor', 2:'Virginica'}

# print()
X=iris["data"]
y=iris["target"]

print(X[:10])
print(y[:10])

feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}

fig,axes = plt.subplots(2, 2, figsize=(12, 6))

for ax,cnt in zip(axes.ravel(), range(4)):
    min_b = math.floor(np.min(X[:, cnt]))
    max_b = math.ceil(np.max(X[:, cnt]))

    bins = np.linspace(min_b, max_b, 25)

    for lab, col in zip(range(0, 3), ('blue', 'red', 'green')):
        ax.hist(X[y == lab, cnt],
                color=col,
                label='class %s' % label_dict[lab],
                bins=bins,
                alpha=0.5, )

    ylims = ax.get_ylim()

    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims) + 2])
    ax.set_xlabel(feature_dict[cnt])
    ax.set_title('Iris histogram #%s' % str(cnt + 1))

    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')

fig.tight_layout()

plt.show()











