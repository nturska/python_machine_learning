import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
from matplotlib import pyplot as plt

kuromasu_five = [[0, 0, 0, 0, '4'],
               [0, '3', 0, '2', 0],
               [0, 0, '2', 0, '2'],
               ['8', 0, 0, 0, 0],
               [0, '3', 0, '2', 0]]


def f(solutions):
    fitness = 0
    for solution in solutions:
        for row in range(len(kuromasu_five)):
            for column in range(len(kuromasu_five[row])):
                if isinstance(kuromasu_five[row][column], str):
                    seen_in_row = []
                    seen_in_column = []
                    for i in range(len(solution)):
                        if row * 5 <= i < (row + 1) * 5:
                            seen_in_row.append(solution[i])
                        if i - column in [0, 5, 10, 15, 20]:
                            seen_in_column.append(solution[i])
                    if (seen_in_row[column] == 1).all():
                        fitness += int(kuromasu_five[row][column])
                    else:
                        seen_in_row[column] = kuromasu_five[row][column]
                        seen_in_column[row] = kuromasu_five[row][column]
                        seen_count1 = 0
                        checked1 = False
                        seen_count2 = 0
                        checked2 = False
                        for i in seen_in_row:
                            if not checked1:
                                if int(i) == 1:
                                    seen_count1 = 0
                                if int(i) == 0:
                                    seen_count1 += 1
                                if isinstance(i, str):
                                    checked1 = True
                                    seen_count1 += 1
                            if checked1:
                                if not i:
                                    seen_count1 += 1
                                if i:
                                    break
                        for i in seen_in_column:
                            if not checked2:
                                if i:
                                    seen_count2 = 0
                                if not i:
                                    seen_count2 += 1
                                if isinstance(i, str):
                                    checked2 = True
                            if checked2:
                                if not i:
                                    seen_count2 += 1
                                if i:
                                    break
                        how_many_is_seen = seen_count1 + seen_count2
                        diff = abs(how_many_is_seen - int(kuromasu_five[row][column]))
                        fitness += diff
    return fitness


options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':2, 'p':1}


optimizer = ps.discrete.BinaryPSO(n_particles=15, dimensions=25, options=options)



# Perform optimization
optimizer.optimize(f, iters=700, verbose=True)
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()
