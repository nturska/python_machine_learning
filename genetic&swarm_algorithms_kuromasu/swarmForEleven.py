import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
from matplotlib import pyplot as plt


kuromasu_eleven = [[0, 0, '9', 0, 0, 0, 0, 0, '8', 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, '7', 0, 0],
                [0, 0, 0, 0, '12', 0, 0, 0, 0, 0, '16'],
                ['9', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, '10', 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, '12', 0, '8', 0, '11', 0, '3', 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, '3', 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '3'],
                ['7', 0, 0, 0, 0, 0, '2', 0, 0, 0, 0],
                [0, 0, '7', 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, '2', 0, 0, 0, 0, 0, '5', 0, 0]]


def f(solutions):
    fitness = 0
    for solution in solutions:
        for row in range(len(kuromasu_eleven)):
            for column in range(len(kuromasu_eleven[row])):
                if isinstance(kuromasu_eleven[row][column], str):
                    seen_in_row = []
                    seen_in_column = []
                    for i in range(len(solution)):
                        if row*11 <= i < (row+1)*11:
                            seen_in_row.append(solution[i])
                        if i-column in [0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110]:
                            seen_in_column.append(solution[i])
                    if seen_in_row[column] == 1 or seen_in_column[row] == 1:
                        fitness += int(kuromasu_eleven[row][column])
                    else:
                        seen_in_row[column] = kuromasu_eleven[row][column]
                        seen_in_column[row] = kuromasu_eleven[row][column]
                        seen_count1 = 0
                        checked1 = False
                        seen_count2 = 0
                        checked2 = False
                        for i in seen_in_row:
                            if not checked1:
                                if i == 1:
                                    seen_count1 = 0
                                if i == 0:
                                    seen_count1 += 1
                                if isinstance(i, str):
                                    checked1 = True
                                    seen_count1 += 1
                            if checked1:
                                if i == 0:
                                    seen_count1 += 1
                                if i == 1:
                                    break
                        for i in seen_in_column:
                            if not checked2:
                                if i == 1:
                                    seen_count2 = 0
                                if i == 0:
                                    seen_count2 += 1
                                if isinstance(i, str):
                                    checked2 = True
                            if checked2:
                                if i == 0:
                                    seen_count2 += 1
                                if i == 1:
                                    break
                        how_many_is_seen = seen_count1+seen_count2
                        diff = abs(how_many_is_seen - int(kuromasu_eleven[row][column]))
                        fitness += diff
    return fitness



options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':2, 'p':1}


optimizer = ps.discrete.BinaryPSO(n_particles=15, dimensions=121, options=options)



# Perform optimization
optimizer.optimize(f, iters=1500, verbose=True)
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()

