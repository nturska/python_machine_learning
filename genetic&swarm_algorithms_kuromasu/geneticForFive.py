import pygad
import time
kuromasu_five = [[0, 0, 0, 0, '4'],
               [0, '3', 0, '2', 0],
               [0, 0, '2', 0, '2'],
               ['8', 0, 0, 0, 0],
               [0, '3', 0, '2', 0]]

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    fitness = 0
    for row in range(len(kuromasu_five)):
        for column in range(len(kuromasu_five[row])):
            if isinstance(kuromasu_five[row][column], str):
                seen_in_row = []
                seen_in_column = []
                for i in range(len(solution)):
                    if row*5 <= i < (row+1)*5:
                        seen_in_row.append(solution[i])
                    if i-column in [0, 5, 10, 15, 20]:
                        seen_in_column.append(solution[i])
                if seen_in_row[column] == 1 or seen_in_column[row] == 1:
                    fitness -= int(kuromasu_five[row][column])
                else:
                    seen_in_row[column] = kuromasu_five[row][column]
                    seen_in_column[row] = kuromasu_five[row][column]
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
                    diff = abs(how_many_is_seen - int(kuromasu_five[row][column]))
                    fitness -= diff
    return fitness


fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 20
num_genes = 25

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 10
num_generations = 700
keep_parents = 4

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 5

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

#uruchomienie algorytmu
start = time.time()
ga_instance.run()
stop = time.time()
print("time of execution ", stop-start)
#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
ga_instance.plot_fitness()