import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from collections import Counter
import warnings
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

warnings.filterwarnings('ignore')

# ===================== text =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# textjoblib（text）text，text：
joblib.parallel_backend('loky', inner_max_num_threads=1)

print(f"text: {SEED} (text)")

# ===================== text =====================
file_path = DATA_DIR / 'ga_training_database.xlsx'
model_path = MODEL_DIR / 'fitness_surrogate_xgboost.pkl'
NUM_COLS = ["LCD (Å)", "PV (cm3/g)", "dist_mean"]
CAT_COLS = ["MC", "topo", "Linker1_SBU", "Linker1_FG", "Linker2_SBU", "Linker2_FG"]
TARGET_COL = "Score"

# text
POP_SIZE = 100
MAX_GEN = 200
SELECTION_RATE = 0.95
CROSSOVER_RATE_MAX = 0.65
CROSSOVER_RATE_MIN = 0.40
MUTATION_RATE_MAX = 0.075
MUTATION_RATE_MIN = 0.05
ADAPTIVE_A = 0.002
ELITE_SIZE = 2

# text
RARE_METAL_THRESHOLD = 20
RARE_METAL_PENALTY = -0.05


# ===================== 1. text =====================
def load_and_preprocess_data(file_path):
    """text"""
    print("text...")
    df = pd.read_excel(file_path)

    required_cols = ['filename'] + NUM_COLS + CAT_COLS + [TARGET_COL]
    available_cols = [col for col in required_cols if col in df.columns]
    df_clean = df[available_cols].copy()

    for col in ['Linker1_SBU', 'Linker1_FG', 'Linker2_SBU', 'Linker2_FG']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype(int)

    for col in ['MC', 'topo']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('unknown')

    print(f"text，text{len(df_clean)}text")
    return df_clean


# ===================== 2. text =====================
def get_feature_bounds(df):
    """text，text1%-99%text"""
    bounds = {}

    for col in NUM_COLS:
        if col in df.columns:
            bounds[col] = {
                'min': df[col].quantile(0.01),
                'max': df[col].quantile(0.99),
                'type': 'continuous'
            }
            print(f"  {col}: [{bounds[col]['min']:.4f}, {bounds[col]['max']:.4f}]")

    for col in CAT_COLS:
        if col in df.columns:
            unique_vals = df[col].dropna().unique().tolist()
            bounds[col] = {
                'values': sorted([v for v in unique_vals if v != 'unknown']),
                'type': 'categorical'
            }
            print(f"  {col}: {len(bounds[col]['values'])}text")

    return bounds


# ===================== 3. text-text =====================
def build_valid_linker_pairs(df):
    """textLinker_SBUtextLinker_FGtext（Linker1textLinker2text）"""
    all_pairs = []

    linker1_pairs = df[['Linker1_SBU', 'Linker1_FG']].dropna()
    linker1_pairs = linker1_pairs[linker1_pairs['Linker1_SBU'] != 0]
    for _, row in linker1_pairs.iterrows():
        all_pairs.append((int(row['Linker1_SBU']), int(row['Linker1_FG'])))

    linker2_pairs = df[['Linker2_SBU', 'Linker2_FG']].dropna()
    linker2_pairs = linker2_pairs[linker2_pairs['Linker2_SBU'] != 0]
    for _, row in linker2_pairs.iterrows():
        all_pairs.append((int(row['Linker2_SBU']), int(row['Linker2_FG'])))

    valid_pairs = list(set(all_pairs))

    print(f"\ntext-text（Linker1textLinker2text）:")
    print(f"  text {len(valid_pairs)} text")

    return valid_pairs


# ===================== 4. text-text =====================
def build_valid_metal_topo_pairs(df):
    """textMCtexttopotext"""
    metal_topo_pairs = df[['MC', 'topo']].dropna()
    metal_topo_pairs = metal_topo_pairs[metal_topo_pairs['MC'] != 'unknown']
    metal_topo_pairs = metal_topo_pairs[metal_topo_pairs['topo'] != 'unknown']

    valid_pairs = []
    for _, row in metal_topo_pairs.iterrows():
        valid_pairs.append((row['MC'], row['topo']))

    valid_pairs = list(set(valid_pairs))

    mc_to_topo = {}
    for mc, topo in valid_pairs:
        if mc not in mc_to_topo:
            mc_to_topo[mc] = []
        mc_to_topo[mc].append(topo)

    topo_to_mc = {}
    for mc, topo in valid_pairs:
        if topo not in topo_to_mc:
            topo_to_mc[topo] = []
        topo_to_mc[topo].append(mc)

    print(f"\ntext-text:")
    print(f"  text {len(valid_pairs)} text")
    print(f"  text {len(mc_to_topo)} text")
    print(f"  text {len(topo_to_mc)} text")

    return valid_pairs, mc_to_topo, topo_to_mc


# ===================== 5. text =====================
def build_topo_ligand_constraint(df):
    """text"""
    topo_constraint = {}

    for topo in df['topo'].unique():
        if topo == 'unknown':
            continue

        topo_df = df[df['topo'] == topo]

        has_single = False
        has_double = False

        for _, row in topo_df.iterrows():
            linker2 = row['Linker2_SBU']
            if pd.isna(linker2) or linker2 == 0:
                has_single = True
            else:
                has_double = True

        if has_single and not has_double:
            topo_constraint[topo] = {'single_only': True, 'double_only': False, 'mixed': False}
        elif has_double and not has_single:
            topo_constraint[topo] = {'single_only': False, 'double_only': True, 'mixed': False}
        else:
            topo_constraint[topo] = {'single_only': False, 'double_only': False, 'mixed': True}

    print(f"\ntext:")
    single_only = [t for t, c in topo_constraint.items() if c['single_only']]
    double_only = [t for t, c in topo_constraint.items() if c['double_only']]
    mixed = [t for t, c in topo_constraint.items() if c['mixed']]
    print(f"  text: {len(single_only)} text")
    print(f"  text: {len(double_only)} text")
    print(f"  text: {len(mixed)} text")

    return topo_constraint


# ===================== 6. text =====================
def calculate_metal_frequencies(df):
    """text"""
    metal_counts = Counter(df['MC'])
    print(f"\ntext:")
    rare_metals = [metal for metal, count in metal_counts.items() if count < RARE_METAL_THRESHOLD]
    print(f"  text（text<{RARE_METAL_THRESHOLD}text）: {len(rare_metals)} text")
    return metal_counts


# ===================== 7. text =====================
def load_model(model_path):
    """textXGBoosttext"""
    print(f"\ntext: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"text: {model_path}")

    model = joblib.load(model_path)
    print("text！")
    return model


# ===================== 8. TAGAtext =====================
class TAGA:
    def __init__(self, model, feature_bounds, valid_linker_pairs,
                 valid_metal_topo_pairs, mc_to_topo, topo_to_mc,
                 topo_ligand_constraint, metal_counts, cat_columns, num_columns):
        self.model = model
        self.feature_bounds = feature_bounds
        self.valid_linker_pairs = valid_linker_pairs
        self.valid_metal_topo_pairs = valid_metal_topo_pairs
        self.mc_to_topo = mc_to_topo
        self.topo_to_mc = topo_to_mc
        self.topo_ligand_constraint = topo_ligand_constraint
        self.metal_counts = metal_counts
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.pop_size = POP_SIZE
        self.max_gen = MAX_GEN
        self.ps = SELECTION_RATE
        self.pc_max = CROSSOVER_RATE_MAX
        self.pc_min = CROSSOVER_RATE_MIN
        self.pm_max = MUTATION_RATE_MAX
        self.pm_min = MUTATION_RATE_MIN
        self.A = ADAPTIVE_A
        self.elite_size = ELITE_SIZE
        self.rare_threshold = RARE_METAL_THRESHOLD
        self.rare_penalty = RARE_METAL_PENALTY

        self.population = None
        self.fitness = None
        self.all_generations_individuals = []
    def get_random_linker_pair(self):
        idx = np.random.randint(0, len(self.valid_linker_pairs))
        return self.valid_linker_pairs[idx]

    def get_random_metal_topo_pair(self):
        idx = np.random.randint(0, len(self.valid_metal_topo_pairs))
        return self.valid_metal_topo_pairs[idx]

    def is_topo_ligand_valid(self, topo, linker2_sbu):
        if topo not in self.topo_ligand_constraint:
            return True
        constraint = self.topo_ligand_constraint[topo]
        is_double = (linker2_sbu != 0)
        if constraint['single_only'] and is_double:
            return False
        if constraint['double_only'] and not is_double:
            return False
        return True

    def is_linker2_consistent(self, sbu, fg):
        """textLinker2textSBUtextFGtext（text0text0）"""
        return (sbu == 0 and fg == 0) or (sbu != 0 and fg != 0)

    def generate_random_individual(self):
        individual = {}

        for col in self.num_columns:
            if col in self.feature_bounds:
                bounds = self.feature_bounds[col]
                individual[col] = np.random.uniform(bounds['min'], bounds['max'])

        mc, topo = self.get_random_metal_topo_pair()
        individual['MC'] = mc
        individual['topo'] = topo

        linker1_sbu, linker1_fg = self.get_random_linker_pair()
        individual['Linker1_SBU'] = linker1_sbu
        individual['Linker1_FG'] = linker1_fg

        constraint = self.topo_ligand_constraint.get(topo, {'mixed': True})

        if constraint['single_only']:
            individual['Linker2_SBU'] = 0
            individual['Linker2_FG'] = 0
        elif constraint['double_only']:
            linker2_sbu, linker2_fg = self.get_random_linker_pair()
            individual['Linker2_SBU'] = linker2_sbu
            individual['Linker2_FG'] = linker2_fg
        else:
            if np.random.random() < 0.5:
                individual['Linker2_SBU'] = 0
                individual['Linker2_FG'] = 0
            else:
                linker2_sbu, linker2_fg = self.get_random_linker_pair()
                individual['Linker2_SBU'] = linker2_sbu
                individual['Linker2_FG'] = linker2_fg

        return individual

    def apply_rare_metal_penalty(self, fitness, population):
        penalized_fitness = fitness.copy()
        for i, ind in enumerate(population):
            metal = ind['MC']
            count = self.metal_counts.get(metal, 0)
            if count < self.rare_threshold:
                penalized_fitness[i] += self.rare_penalty
        return penalized_fitness

    def get_structure_signature(self, individual):
        return (
            individual['MC'],
            individual['topo'],
            individual['Linker1_SBU'],
            individual['Linker1_FG'],
            individual['Linker2_SBU'],
            individual['Linker2_FG']
        )

    def get_non_duplicate_best(self, population, fitness, existing_structures):
        sorted_indices = np.argsort(fitness)[::-1]
        for idx in sorted_indices:
            signature = self.get_structure_signature(population[idx])
            if signature not in existing_structures:
                return idx, population[idx], fitness[idx]
        return sorted_indices[0], population[sorted_indices[0]], fitness[sorted_indices[0]]

    def is_valid_linker_pair(self, sbu, fg):
        if sbu == 0 and fg == 0:
            return True
        return (sbu, fg) in self.valid_linker_pairs

    def is_valid_metal_topo_pair(self, mc, topo):
        return (mc, topo) in self.valid_metal_topo_pairs

    def repair_all_constraints(self, individual):
        # text-text
        if not self.is_valid_metal_topo_pair(individual['MC'], individual['topo']):
            if individual['MC'] in self.mc_to_topo and self.mc_to_topo[individual['MC']]:
                individual['topo'] = np.random.choice(self.mc_to_topo[individual['MC']])
            else:
                mc, topo = self.get_random_metal_topo_pair()
                individual['MC'] = mc
                individual['topo'] = topo

        # textLinker2textSBUtextFGtext
        if not self.is_linker2_consistent(individual.get('Linker2_SBU', 0), individual.get('Linker2_FG', 0)):
            individual['Linker2_SBU'] = 0
            individual['Linker2_FG'] = 0

        # text-text
        topo = individual['topo']
        linker2_sbu = individual.get('Linker2_SBU', 0)

        if topo in self.topo_ligand_constraint:
            constraint = self.topo_ligand_constraint[topo]
            is_double = (linker2_sbu != 0)

            if constraint['single_only'] and is_double:
                individual['Linker2_SBU'] = 0
                individual['Linker2_FG'] = 0
            elif constraint['double_only'] and not is_double:
                linker2_sbu, linker2_fg = self.get_random_linker_pair()
                individual['Linker2_SBU'] = linker2_sbu
                individual['Linker2_FG'] = linker2_fg

        # textLinker1
        if individual['Linker1_SBU'] != 0:
            if not self.is_valid_linker_pair(individual['Linker1_SBU'], individual['Linker1_FG']):
                sbu, fg = self.get_random_linker_pair()
                individual['Linker1_SBU'] = sbu
                individual['Linker1_FG'] = fg

        # textLinker2（text）
        if individual.get('Linker2_SBU', 0) != 0:
            if not self.is_valid_linker_pair(individual['Linker2_SBU'], individual['Linker2_FG']):
                sbu, fg = self.get_random_linker_pair()
                individual['Linker2_SBU'] = sbu
                individual['Linker2_FG'] = fg

        return individual

    def initialize_population(self, initial_individuals, existing_structures):
        print(f"\ntext，text: {len(initial_individuals)}")
        self.population = []
        added_count = 0
        repared_count = 0

        for ind_dict in initial_individuals:
            need_repair = False

            if not self.is_valid_metal_topo_pair(ind_dict['MC'], ind_dict['topo']):
                mc, topo = self.get_random_metal_topo_pair()
                ind_dict['MC'] = mc
                ind_dict['topo'] = topo
                need_repair = True

            # textLinker2text
            if not self.is_linker2_consistent(ind_dict.get('Linker2_SBU', 0), ind_dict.get('Linker2_FG', 0)):
                ind_dict['Linker2_SBU'] = 0
                ind_dict['Linker2_FG'] = 0
                need_repair = True

            topo = ind_dict['topo']
            linker2_sbu = ind_dict.get('Linker2_SBU', 0)
            if topo in self.topo_ligand_constraint:
                constraint = self.topo_ligand_constraint[topo]
                is_double = (linker2_sbu != 0)
                if constraint['single_only'] and is_double:
                    ind_dict['Linker2_SBU'] = 0
                    ind_dict['Linker2_FG'] = 0
                    need_repair = True
                elif constraint['double_only'] and not is_double:
                    sbu, fg = self.get_random_linker_pair()
                    ind_dict['Linker2_SBU'] = sbu
                    ind_dict['Linker2_FG'] = fg
                    need_repair = True

            if ind_dict['Linker1_SBU'] != 0:
                if not self.is_valid_linker_pair(ind_dict['Linker1_SBU'], ind_dict['Linker1_FG']):
                    sbu, fg = self.get_random_linker_pair()
                    ind_dict['Linker1_SBU'] = sbu
                    ind_dict['Linker1_FG'] = fg
                    need_repair = True

            if ind_dict.get('Linker2_SBU', 0) != 0:
                if not self.is_valid_linker_pair(ind_dict['Linker2_SBU'], ind_dict['Linker2_FG']):
                    sbu, fg = self.get_random_linker_pair()
                    ind_dict['Linker2_SBU'] = sbu
                    ind_dict['Linker2_FG'] = fg
                    need_repair = True

            if need_repair:
                repared_count += 1

            signature = self.get_structure_signature(ind_dict)
            if signature not in existing_structures:
                self.population.append(ind_dict.copy())
                existing_structures.add(signature)
                added_count += 1

        print(f"  text {added_count} text（text {repared_count} text）")

        while len(self.population) < self.pop_size:
            random_ind = self.generate_random_individual()
            signature = self.get_structure_signature(random_ind)
            if signature not in existing_structures:
                self.population.append(random_ind)
                existing_structures.add(signature)

        print(f"  text: {len(self.population)}")
        return self.population

    def evaluate_fitness(self, population):
        df_pop = pd.DataFrame(population)

        for col in self.num_columns + self.cat_columns:
            if col not in df_pop.columns:
                if col in self.num_columns:
                    bounds = self.feature_bounds[col]
                    df_pop[col] = (bounds['min'] + bounds['max']) / 2
                else:
                    df_pop[col] = self.feature_bounds[col]['values'][0]

        try:
            fitness = self.model.predict(df_pop[self.num_columns + self.cat_columns])
        except Exception as e:
            print(f"text: {e}")
            fitness = np.zeros(len(population))

        fitness = self.apply_rare_metal_penalty(fitness, population)
        return fitness

    def selection(self, fitness):
        selected_indices = []
        n_pop = len(self.population)

        for _ in range(n_pop - self.elite_size):
            candidates_idx = np.random.choice(n_pop, 4, replace=False)
            candidates_fitness = fitness[candidates_idx]
            sorted_idx = candidates_idx[np.argsort(candidates_fitness)[::-1]]

            if np.random.random() < self.ps:
                selected_indices.append(sorted_idx[0])
            else:
                selected_indices.append(sorted_idx[1])

        return selected_indices

    def crossover(self, parent1, parent2, pc):
        if np.random.random() < pc:
            keys = list(parent1.keys())
            crossover_point = np.random.randint(1, len(keys))

            child1 = {}
            child2 = {}

            for i, key in enumerate(keys):
                if i < crossover_point:
                    child1[key] = parent1[key]
                    child2[key] = parent2[key]
                else:
                    child1[key] = parent2[key]
                    child2[key] = parent1[key]

            child1 = self.repair_all_constraints(child1)
            child2 = self.repair_all_constraints(child2)

            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual, pm):
        if np.random.random() < pm:
            keys = list(individual.keys())
            mutate_key = np.random.choice(keys)

            if mutate_key in self.num_columns:
                bounds = self.feature_bounds[mutate_key]
                sigma = 0.1 * (bounds['max'] - bounds['min'])
                new_value = individual[mutate_key] + np.random.normal(0, sigma)
                individual[mutate_key] = np.clip(new_value, bounds['min'], bounds['max'])

            elif mutate_key == 'MC':
                current_topo = individual['topo']
                if current_topo in self.topo_to_mc:
                    available_mc = [mc for mc in self.topo_to_mc[current_topo]
                                    if mc != individual['MC']]
                    if available_mc:
                        individual['MC'] = np.random.choice(available_mc)
                if individual['MC'] == individual.get('mc_before', ''):
                    mc, topo = self.get_random_metal_topo_pair()
                    individual['MC'] = mc
                    individual['topo'] = topo

            elif mutate_key == 'topo':
                current_mc = individual['MC']
                if current_mc in self.mc_to_topo:
                    available_topo = [t for t in self.mc_to_topo[current_mc]
                                      if t != individual['topo']]
                    if available_topo:
                        individual['topo'] = np.random.choice(available_topo)
                if individual['topo'] == individual.get('topo_before', ''):
                    mc, topo = self.get_random_metal_topo_pair()
                    individual['MC'] = mc
                    individual['topo'] = topo

            elif mutate_key in ['Linker1_SBU', 'Linker1_FG']:
                sbu, fg = self.get_random_linker_pair()
                individual['Linker1_SBU'] = sbu
                individual['Linker1_FG'] = fg

            elif mutate_key in ['Linker2_SBU', 'Linker2_FG']:
                topo = individual['topo']
                constraint = self.topo_ligand_constraint.get(topo, {'mixed': True})

                if constraint['single_only']:
                    individual['Linker2_SBU'] = 0
                    individual['Linker2_FG'] = 0
                elif constraint['double_only']:
                    sbu, fg = self.get_random_linker_pair()
                    individual['Linker2_SBU'] = sbu
                    individual['Linker2_FG'] = fg
                else:
                    if np.random.random() < 0.3:
                        individual['Linker2_SBU'] = 0
                        individual['Linker2_FG'] = 0
                    else:
                        sbu, fg = self.get_random_linker_pair()
                        individual['Linker2_SBU'] = sbu
                        individual['Linker2_FG'] = fg

        return individual

    def get_adaptive_rates(self, gen, fitness_values):
        avg_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        min_fitness = np.min(fitness_values)

        if max_fitness == min_fitness:
            pc = (self.pc_max + self.pc_min) / 2
            pm = (self.pm_max + self.pm_min) / 2
            return pc, pm

        f_prime = np.random.choice(fitness_values)
        f = np.random.choice(fitness_values)

        pc = self.pc_min + (self.pc_max - self.pc_min) * np.exp(
            -self.A * abs(f_prime - avg_fitness) / (max_fitness - min_fitness))
        pm = self.pm_min + (self.pm_max - self.pm_min) * np.exp(
            -self.A * abs(f - avg_fitness) / (max_fitness - min_fitness))
        pm = pm * (1 - gen / self.max_gen * 0.3)

        return pc, pm

    def evolve(self, initial_population_dicts, existing_structures_set):
        print("\ntext...")

        self.initialize_population(initial_population_dicts, existing_structures_set)

        generation_info = []

        for gen in range(self.max_gen):
            fitness = self.evaluate_fitness(self.population)
            self.fitness = fitness

            best_idx, best_individual, best_fitness = self.get_non_duplicate_best(
                self.population, fitness, existing_structures_set
            )

            generation_info.append({
                'generation': gen + 1,
                'best_score': best_fitness,
                'avg_score': np.mean(fitness),
                'max_score': np.max(fitness),
                'min_score': np.min(fitness),
                'std_score': np.std(fitness),
                'best_individual': best_individual
            })
            if not hasattr(self, 'all_generations_individuals'):
                self.all_generations_individuals = []

            generation_individuals = []
            for i, ind in enumerate(self.population):
                ind_record = {
                    'generation': gen + 1,
                    'individual_id': i + 1,
                    'fitness': fitness[i],
                    'LCD (Å)': ind.get('LCD (Å)', None),
                    'PV (cm3/g)': ind.get('PV (cm3/g)', None),
                    'dist_mean': ind.get('dist_mean', None),
                    'MC': ind.get('MC', None),
                    'topo': ind.get('topo', None),
                    'Linker1_SBU': ind.get('Linker1_SBU', None),
                    'Linker1_FG': ind.get('Linker1_FG', None),
                    'Linker2_SBU': ind.get('Linker2_SBU', None),
                    'Linker2_FG': ind.get('Linker2_FG', None)
                }
                generation_individuals.append(ind_record)

            self.all_generations_individuals.extend(generation_individuals)
            print(f"text{gen + 1:3d}text | textScore: {best_fitness:.4f} | "
                  f"textScore: {np.mean(fitness):.4f} | text: {np.std(fitness):.4f}")

            selected_indices = self.selection(fitness)
            selected_pop = [self.population[i] for i in selected_indices]

            pc, pm = self.get_adaptive_rates(gen, fitness)

            new_population = []

            elite_indices = np.argsort(fitness)[-self.elite_size:]
            for idx in elite_indices:
                if self.get_structure_signature(self.population[idx]) not in existing_structures_set:
                    new_population.append(self.population[idx].copy())

            while len(new_population) < self.pop_size:
                parent1_idx = np.random.choice(len(selected_pop))
                parent2_idx = np.random.choice(len(selected_pop))
                while parent2_idx == parent1_idx and len(selected_pop) > 1:
                    parent2_idx = np.random.choice(len(selected_pop))

                parent1 = selected_pop[parent1_idx].copy()
                parent2 = selected_pop[parent2_idx].copy()

                child1, child2 = self.crossover(parent1, parent2, pc)
                child1 = self.mutate(child1, pm)
                child2 = self.mutate(child2, pm)

                if self.get_structure_signature(child1) not in existing_structures_set:
                    new_population.append(child1)
                if len(new_population) < self.pop_size and self.get_structure_signature(
                        child2) not in existing_structures_set:
                    new_population.append(child2)

            self.population = new_population[:self.pop_size]

        final_fitness = self.evaluate_fitness(self.population)
        final_best_idx, final_best_individual, final_best_fitness = self.get_non_duplicate_best(
            self.population, final_fitness, existing_structures_set
        )

        print(f"\ntext！textScore: {final_best_fitness:.4f}")

        return generation_info


# ===================== 9. text =====================
def main():
    df = load_and_preprocess_data(file_path)

    print("\ntext（1%-99%text）:")
    feature_bounds = get_feature_bounds(df)

    valid_linker_pairs = build_valid_linker_pairs(df)

    valid_metal_topo_pairs, mc_to_topo, topo_to_mc = build_valid_metal_topo_pairs(df)

    topo_ligand_constraint = build_topo_ligand_constraint(df)

    metal_counts = calculate_metal_frequencies(df)

    df_sorted = df.sort_values(TARGET_COL, ascending=False)
    initial_population_df = df_sorted.head(POP_SIZE).copy()

    print(f"\ntext: text{POP_SIZE}textScoretext")
    print(f"Scoretext: {initial_population_df[TARGET_COL].min():.4f} - {initial_population_df[TARGET_COL].max():.4f}")

    model = load_model(model_path)

    initial_population_dicts = []
    for _, row in initial_population_df.iterrows():
        individual = {}
        for col in NUM_COLS + CAT_COLS:
            if col in row:
                individual[col] = row[col]
        initial_population_dicts.append(individual)

    existing_structures_set = set()
    for _, row in df.iterrows():
        signature = (
            row['MC'],
            row['topo'],
            int(row['Linker1_SBU']) if pd.notna(row['Linker1_SBU']) else 0,
            int(row['Linker1_FG']) if pd.notna(row['Linker1_FG']) else 0,
            int(row['Linker2_SBU']) if pd.notna(row['Linker2_SBU']) else 0,
            int(row['Linker2_FG']) if pd.notna(row['Linker2_FG']) else 0
        )
        existing_structures_set.add(signature)
    print(f"\ntext: {len(existing_structures_set)}")

    taga = TAGA(
        model=model,
        feature_bounds=feature_bounds,
        valid_linker_pairs=valid_linker_pairs,
        valid_metal_topo_pairs=valid_metal_topo_pairs,
        mc_to_topo=mc_to_topo,
        topo_to_mc=topo_to_mc,
        topo_ligand_constraint=topo_ligand_constraint,
        metal_counts=metal_counts,
        cat_columns=CAT_COLS,
        num_columns=NUM_COLS
    )

    generation_info = taga.evolve(initial_population_dicts, existing_structures_set)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(taga, 'all_generations_individuals') and taga.all_generations_individuals:
        df_all_individuals = pd.DataFrame(taga.all_generations_individuals)
        all_individuals_file = os.path.join(output_dir, "text.xlsx")
        df_all_individuals.to_excel(all_individuals_file, index=False)
        print(f"text: {all_individuals_file}")
        print(f"  text {len(df_all_individuals)} text（{taga.max_gen}text × {taga.pop_size}text）")

    generation_records = []
    for info in generation_info:
        ind = info['best_individual']
        record = {
            'generation': info['generation'],
            'Score': info['best_score'],
            'avg_score': info['avg_score'],
            'max_score': info['max_score'],
            'min_score': info['min_score'],
            'std_score': info['std_score']
        }
        for col in NUM_COLS + CAT_COLS:
            record[col] = ind.get(col, None)
        generation_records.append(record)

    df_generation = pd.DataFrame(generation_records)
    gen_file = os.path.join(output_dir, "text.xlsx")
    df_generation.to_excel(gen_file, index=False)
    print(f"\ntext: {gen_file}")

    final_fitness = taga.evaluate_fitness(taga.population)

    unique_final = []
    unique_signatures = set()
    for idx in range(len(taga.population)):
        signature = taga.get_structure_signature(taga.population[idx])
        if signature not in unique_signatures:
            unique_signatures.add(signature)
            unique_final.append((idx, final_fitness[idx], taga.population[idx]))

    unique_final.sort(key=lambda x: x[1], reverse=True)
    unique_final = unique_final[:50]

    final_records = []
    for rank, (idx, score, ind) in enumerate(unique_final, 1):
        record = {
            'rank': rank,
            'Score': score
        }
        for col in NUM_COLS + CAT_COLS:
            record[col] = ind.get(col, None)
        final_records.append(record)

    df_final = pd.DataFrame(final_records)
    final_file = os.path.join(output_dir, "textTOP50.xlsx")
    df_final.to_excel(final_file, index=False)
    print(f"textTOP50text: {final_file}")

    df_stats = pd.DataFrame({
        'generation': [info['generation'] for info in generation_info],
        'best_score': [info['best_score'] for info in generation_info],
        'avg_score': [info['avg_score'] for info in generation_info],
        'max_score': [info['max_score'] for info in generation_info],
        'min_score': [info['min_score'] for info in generation_info],
        'std_score': [info['std_score'] for info in generation_info]
    })
    stats_file = os.path.join(output_dir, "text.xlsx")
    df_stats.to_excel(stats_file, index=False)

    print("\n" + "=" * 60)
    print("text:")
    print("=" * 60)
    best = final_records[0]
    for key, value in best.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("text:")
    print("=" * 60)
    print(f"textScore: {generation_info[0]['best_score']:.4f}")
    print(f"textScore: {generation_info[-1]['best_score']:.4f}")
    print(f"text: {generation_info[-1]['best_score'] - generation_info[0]['best_score']:.4f}")
    print(f"text: {generation_info[-1]['best_score'] - generation_info[0]['best_score']:.4f}")
    print(f"textScore: {generation_info[0]['avg_score']:.4f}")
    print(f"textScore: {generation_info[-1]['avg_score']:.4f}")

    print(f"\ntext: {output_dir}")

    return df_generation, df_final


if __name__ == "__main__":
    df_gen, df_final = main()
