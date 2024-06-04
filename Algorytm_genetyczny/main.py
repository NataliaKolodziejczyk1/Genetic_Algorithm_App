import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import random
import threading


class Individual:
    def __init__(self,num):
        self.len = num
        self.chromosome = np.zeros(self.len)
        self.fitness = np.Inf

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome

    def random_chromosome(self):
        self.chromosome = np.random.permutation(self.len)

    def calculate_fitness(self,distance_table):
        self.fitness = 0
        for i in range(self.len-1):
            self.fitness += distance_table.iloc[self.chromosome[i],self.chromosome[i+1]]
        self.fitness += distance_table.iloc[self.chromosome[self.len-1],self.chromosome[0]]

    def crossover(self,parent2):
        start_locus = random.randint(0,self.len-1)
        end_locus = random.randint(start_locus,self.len-1)
        child1 = Individual(self.len)
        child1.set_chromosome(self.chromosome.copy())
        child2 = Individual(self.len)
        child2.set_chromosome(parent2.chromosome.copy())
        part1 = self.chromosome[start_locus:end_locus+1]
        part2 = parent2.chromosome[start_locus:end_locus + 1]
        # part that we must change in chlid1 and give it to child2
        child1_change = set(part2).difference(set(part1))
        child2_change = set(part1).difference(set(part2))
        for i in range(self.len):
            if child1.chromosome[i] in child1_change:
                for j in range(self.len):
                    if child2.chromosome[j] in child2_change:
                        temp = child1.chromosome[i]
                        child1.chromosome[i] = child2.chromosome[j]
                        child2.chromosome[j] = temp
        child1.chromosome[start_locus:end_locus+1] = part2
        child2.chromosome[start_locus:end_locus+1] = part1
        return [child1, child2]

    def mutate(self):
        # locus we want to move
        n = random.randint(1,self.len-1)
        # where we move it
        m = random.randint(0,n-1)
        gene = self.chromosome[n]
        new_chromosome = np.delete(self.chromosome, n)
        new_chromosome = np.insert(new_chromosome, m, gene)
        self.set_chromosome(new_chromosome)





class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Genetic Algorithm')
        self.geometry("700x500")

        self.cities = np.array([[], []])
        self.num = 0

        frame1 = self.create_frame(tk.RIGHT,tk.X,'#BCAC9B',10,10)
        frame2 = self.create_frame(tk.TOP, tk.X, 'white', 0, 0)

        E1 = self.create_label_entry(frame1, "Number of Cities:", 0, "10")
        E2 = self.create_label_entry(frame1, "Crossover Rate:", 2, "0.9")
        E3 = self.create_label_entry(frame1, "Mutation Rate:", 3, "0.05")
        E4 = self.create_label_entry(frame1, "Population size:", 4, "100")
        E5 = self.create_label_entry(frame1, "Max number of iterations:", 5, "10000")

        self.draw_best_solution_var = tk.BooleanVar()
        self.dont_draw_realtime = tk.BooleanVar()
        self.create_checkbutton(frame1, 9,"Draw only best solution", self.draw_best_solution_var)
        self.create_checkbutton(frame1, 10, "No realtime drawing", self.dont_draw_realtime)

        options1 = ["total replacement","elitist"]
        self.option_var1 = tk.StringVar(self)
        self.option_var1.set(options1[0])
        self.create_dropmenu(frame1,7,0,"Succession",options1,self.option_var1)

        options2 = ["rank selection", "roulette selection"]
        self.option_var2 = tk.StringVar(self)
        self.option_var2.set(options2[0])
        self.create_dropmenu(frame1, 7, 1, "Selection", options2, self.option_var2)

        self.crossover_rate = float(E2.get())
        self.mutation_rate = float(E3.get())
        self.population_size = int(E4.get())
        self.max_iter = int(E5.get())

        fig, ax = plt.subplots(2,1,figsize=(10, 20))
        ax[1].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        canvas = FigureCanvasTkAgg(fig, master=frame2)
        canvas.get_tk_widget().pack(side=tk.TOP)
        self.draw_cities(ax[0], canvas)

        self.create_button(frame1,"Init Cities",1,self.cities_button_action,ax,canvas,E1)
        self.create_button(frame1, "START", 8, self.start_button_action, E2, E3, E4,E5,ax,canvas)

        self.label_generation = self.create_label(frame1,12, "Current Generation: 0")
    def create_frame(self,side,fill,color,padx,pady):
        frame = tk.Frame(self,bg=color)
        frame.pack(side = side, fill = fill, padx = padx, pady = pady)
        return frame

    def create_label_entry(self,frame,text,row,placeholder):
        L = tk.Label(frame, text=text, bg='white')
        L.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        E = tk.Entry(frame, bd=5)
        E.insert(0, placeholder)
        E.grid(row=row, column=1, padx=5, pady=5, sticky="e")
        return E

    def create_label(self, frame, row, text):
        label = tk.Label(frame, text=text, bg='white',padx=10,pady=10)
        label.grid(row=row, column=0, columnspan=2, padx=20, pady=20)
        return label
    def create_button(self,frame,text,row,func,*args):
        button = tk.Button(frame, text=text, width=15)
        button.config(command=lambda: func(*args))
        button.grid(row=row, column=0, columnspan=2, padx=5, pady=5)

    def create_checkbutton(self, frame, row, text, variable):
        checkbutton = tk.Checkbutton(frame, text=text, variable=variable,bg='white')
        checkbutton.grid(row=row, column=0, columnspan=2, padx=5, pady=5)

    def create_dropmenu(self,frame,row,column,text,options,variable):
        label_menu = tk.Label(frame, text=text,pady=5,padx=10)
        label_menu.grid(row=row-1, column=column)

        dropdown_menu = tk.OptionMenu(frame, variable, *options)
        dropdown_menu.grid(row=row, column=column,padx=5,pady=5)
    def draw_cities(self,ax, canvas):
        ax.clear()
        ax.axis("square")
        ax.set_xlim((0, 100))
        ax.set_ylim((0, 100))
        ax.scatter(self.cities[0], self.cities[1], marker="o", s=6,color="#4F3130")

    def cities_button_action(self, ax, canvas, e):
        ax[1].clear()
        ax[1].get_yaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        self.label_generation.config(text=f"Current Generation: 0")
        self.num = int(e.get())
        self.cities = np.random.rand(2,self.num)*100
        self.draw_cities(ax[0],canvas)
        canvas.draw()

    def start_button_action(self,e1, e2, e3,e4,ax,canvas):
        self.crossover_rate = float(e1.get())
        self.mutation_rate = float(e2.get())
        self.population_size = int(e3.get())
        self.max_iter = int(e4.get())
        print(self.crossover_rate)
        print(self.mutation_rate)
        print(self.population_size)
        print(self.max_iter)
        print(self.draw_best_solution_var.get())
        print(self.dont_draw_realtime.get())
        print(self.option_var1.get())
        print(self.option_var2.get())
        thread = threading.Thread(target=self.start_genetic_algorithm, args=(ax, canvas))
        thread.start()

    def start_genetic_algorithm(self,ax,canvas):
        pop_without_improvement = 0
        max_pop_without_improvement = 0
        succesion = self.option_var1.get()
        selection = self.option_var2.get()
        if succesion == "elitist" and selection == "rank selection":
            max_pop_without_improvement = 150
        if succesion == "elitist" and selection == "roulette selection":
            max_pop_without_improvement = 100
        if succesion == "total replacement" and selection == "roulette selection":
            max_pop_without_improvement = 10
        if succesion == "total replacement" and selection == "rank selection":
            max_pop_without_improvement = 20

        alg = GeneticAlgorithm(self.cities, self.num, self.crossover_rate, self.mutation_rate,
                               self.population_size,self.max_iter,succesion,selection)
        # Populacja początkowa
        alg.initialize_population()
        alg.count_distance_table()
        # Ocena wszystkich osobników populacji
        alg.mark_the_population()

        self.draw_cities(ax[0], canvas)
        ax[1].clear()
        canvas.draw()
        ax[1].get_yaxis().set_visible(True)
        ax[1].get_xaxis().set_visible(True)

        if not self.dont_draw_realtime.get():
            if not self.draw_best_solution_var.get():
                self.draw_solutions(ax[0], canvas, alg)
            self.draw_best_solution(ax[0], canvas, alg)
            ax[1].plot(alg.best_solutions, color='#D33F49')
            canvas.draw()

        while True:
            print(alg.current_generation)
            alg.current_generation += 1
            self.label_generation.config(text=f"Current Generation: {alg.current_generation}")
            alg.selection()
            alg.crossover()
            alg.mutation()
            alg.succession()
            alg.mark_the_population()
            if alg.best_solutions[alg.current_generation-1] == alg.best_solutions[alg.current_generation]:
                pop_without_improvement += 1
            else:
                pop_without_improvement = 0
            if pop_without_improvement == max_pop_without_improvement:
                break
            if alg.max_iterations <= alg.current_generation:
                break
            if not self.dont_draw_realtime.get():
                self.draw_cities(ax[0], canvas)
                if not self.draw_best_solution_var.get():
                    self.draw_solutions(ax[0], canvas, alg)
                self.draw_best_solution(ax[0], canvas, alg)
                ax[1].plot(alg.best_solutions,color = '#D33F49')
                canvas.draw()
        print(alg.best_solutions)
        if self.dont_draw_realtime.get():
            ax[1].clear()
            ax[1].get_yaxis().set_visible(True)
            ax[1].get_xaxis().set_visible(True)
            self.draw_cities(ax[0], canvas)
            if not self.draw_best_solution_var.get():
                self.draw_solutions(ax[0], canvas, alg)
            self.draw_best_solution(ax[0], canvas, alg)
            ax[1].plot(alg.best_solutions, color='#D33F49')
            canvas.draw()
        plt.savefig('plot.jpg')


    def draw_solutions(self, ax, canvas,alg):
        for i in range(5 if alg.population_size > 5 else alg.population_size):
            x = np.array(alg.cities['X'][list(alg.population[i].chromosome)])
            y = np.array(alg.cities['Y'][list(alg.population[i].chromosome)])
            x = np.append(x,alg.cities['X'][alg.population[i].chromosome[0]])
            y = np.append(y,alg.cities['Y'][alg.population[i].chromosome[0]])
            ax.plot(x, y,alpha=0.2,color='#AA5042')
            #canvas.draw()

    def draw_best_solution(self, ax, canvas, alg):
        x = np.array(alg.cities['X'][list(alg.population[0].chromosome)])
        y = np.array(alg.cities['Y'][list(alg.population[0].chromosome)])
        x = np.append(x, alg.cities['X'][alg.population[0].chromosome[0]])
        y = np.append(y, alg.cities['Y'][alg.population[0].chromosome[0]])
        ax.plot(x, y, alpha=0.9, color='#753742')
        #canvas.draw()

    def tksleep(self,t):
        'emulating time.sleep(seconds)'
        ms = int(t * 1000)
        root = tk._get_default_root('sleep')
        var = tk.IntVar(root)
        root.after(ms, var.set, 1)
        root.wait_variable(var)
class GeneticAlgorithm():
    def __init__(self,cities,num,crossover_rate, mutation_rate, population_size,max_iter,succesion,selection):
        self.cities = pd.DataFrame({"X":cities[0],"Y":cities[1]})
        self.num = num
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.population = list(range(population_size))
        self.distance_table = self.count_distance_table()
        self.fitness = np.zeros(self.population_size)
        self.best_solutions = np.array([])
        self.parents = list(range(population_size))
        self.children = list(range(population_size))
        self.max_iterations = max_iter
        self.current_generation = 0
        self.succesion_type = succesion
        self.selection_type = selection
    def initialize_population(self):
        for i in range(self.population_size):
            self.population[i] = Individual(self.num)
            self.population[i].random_chromosome()

    def count_distance_table(self):
        distance_table = np.zeros((self.num,self.num))
        for i in range(self.num):
            distance_table[i,:] = ((self.cities['X'][i] - self.cities['X'])**2+(self.cities['Y'][i] - self.cities['Y'])**2)**0.5
        return pd.DataFrame(distance_table)

    def mark_the_population(self):
        for i in range(self.population_size):
            self.population[i].calculate_fitness(self.distance_table)
            self.fitness[i] = self.population[i].fitness
        ranking = sorted([*zip(list(self.fitness), self.population)],key=lambda el: el[0])
        self.fitness = np.array([x for x,y in ranking])
        self.population = [y for x, y in ranking]
        self.best_solutions = np.append(self.best_solutions,self.population[0].fitness)

    def rank_selection(self):
        ranks = 1/self.population_size*(1.5-np.arange(self.population_size)/(self.population_size-1)).cumsum()
        parent_population = []
        for i in range(self.population_size):
            r = random.random()
            parent_population.append(np.array(self.population)[(ranks > r).cumsum() == 1][0])
        self.parents = parent_population

    def rullete_selection(self):
        fitness_sum = sum(1/self.fitness)
        prob = (1/self.fitness/fitness_sum).cumsum()
        parent_population = []
        for i in range(self.population_size):
            r = random.random()
            parent_population.append(np.array(self.population)[(prob > r).cumsum() == 1][0])
        self.parents = parent_population

    def selection(self):
        if self.selection_type == "rank selection":
            self.rank_selection()
        if self.selection_type == "roulette selection":
            self.rullete_selection()

    def crossover(self):
        n = 1 if self.population_size % 2 else 0
        for i in range(n,self.population_size-1,2):
            if random.random() < self.crossover_rate:
                children = self.parents[i].crossover(self.parents[i+1])
                self.children[i] = children[0]
                self.children[i+1] = children[1]
            else:
                self.children[i] = Individual(self.num)
                self.children[i].set_chromosome(self.parents[i].chromosome)
                self.children[i].calculate_fitness(self.distance_table)
                self.children[i + 1] = Individual(self.num)
                self.children[i+1].set_chromosome(self.parents[i+1].chromosome)
                self.children[i+1].calculate_fitness(self.distance_table)
        if n == 1:
            self.children[0] = Individual(self.num)
            self.children[0].set_chromosome(self.parents[0].chromosome)
            self.children[0].calculate_fitness(self.distance_table)

    def succession(self):
        if self.succesion_type == "total replacement":
            self.population = self.children
        if self.succesion_type == "elitist":
            self.population[1:] = self.children[1:]

    def mutation(self):
        for i in range(self.population_size):
            if random.random() < self.mutation_rate:
                self.children[i].mutate()







if __name__ == '__main__':
    app = Application()
    app.mainloop()
    exit()



