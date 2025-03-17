import random

tasks = list(range(10))
cpu_time = [random.randint(80, 150) for _ in tasks]
gpu_time = [t // 4 + random.randint(-5, 5) for t in cpu_time]

pop_size = 20
generations = 50
mutation_rate = 0.1

def random_schedule():
    return [random.choice([0, 1]) for _ in tasks]

population = [random_schedule() for _ in range(pop_size)]

def schedule_makespan(schedule):
    total_cpu = total_gpu = 0
    for j, assign in enumerate(schedule):
        if assign == 0:
            total_cpu += cpu_time[j]
        else:
            total_gpu += gpu_time[j]
    return max(total_cpu, total_gpu)

for gen in range(generations):
    population = sorted(population, key=schedule_makespan)
    best = population[0]
    best_cost = schedule_makespan(best)
    if gen % 10 == 0:
        print(f"Generation {gen}: Best makespan = {best_cost} ms (CPU={sum(cpu_time[j] for j in range(len(tasks)) if best[j]==0)} ms, GPU={sum(gpu_time[j] for j in range(len(tasks)) if best[j]==1)} ms)")
    survivors = population[:pop_size//2]
    offspring = []
    while len(offspring) < pop_size//2:
        parents = random.sample(survivors, 2)
        cut = random.randint(1, len(tasks)-1)
        child = parents[0][:cut] + parents[1][cut:]
        offspring.append(child)
    for child in offspring:
        if random.random() < mutation_rate:
            idx = random.randrange(len(tasks))
            child[idx] = 1 - child[idx]
    population = survivors + offspring

best_schedule = population[0]
print("Final best schedule assignment (0=CPU,1=GPU):", best_schedule)
print("CPU tasks:", [j for j in tasks if best_schedule[j]==0])
print("GPU tasks:", [j for j in tasks if best_schedule[j]==1])
print("Final makespan:", schedule_makespan(best_schedule), "ms")
