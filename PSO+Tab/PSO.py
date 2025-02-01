import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # 진행도 표시용

class PSOFeatureSelection:
    def __init__(self, num_features, num_particles=20, max_iter=50, inertia_weight=0.7, c1=1.5, c2=2.0):
        self.num_features = num_features
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2

        self.particles = np.random.randint(0, 2, (num_particles, num_features))
        self.velocities = np.random.uniform(-1, 1, (num_particles, num_features))

        self.personal_best = np.copy(self.particles)
        self.personal_best_score = np.full(num_particles, -np.inf)

        self.global_best = None
        self.global_best_score = -np.inf

    def evaluate_fitness(self, X, y, model, particle):
        selected_features = np.where(particle == 1)[0]
        if len(selected_features) == 0:
            return 0
        
        X_selected = X[:, selected_features]
        X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)

    def optimize(self, X, y, model):
        for iter_num in tqdm(range(self.max_iter), desc="PSO 진행 중", unit="iter"):
            for i in range(self.num_particles):
                fitness = self.evaluate_fitness(X, y, model, self.particles[i])
                
                if fitness > self.personal_best_score[i]:
                    self.personal_best[i] = self.particles[i]
                    self.personal_best_score[i] = fitness

                if fitness > self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best = np.copy(self.particles[i])

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = self.c1 * r1 * (self.personal_best[i] - self.particles[i])
                social = self.c2 * r2 * (self.global_best - self.particles[i]) if self.global_best is not None else 0
                
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive + social

                sigmoid = 1 / (1 + np.exp(-self.velocities[i]))
                self.particles[i] = (np.random.rand(self.num_features) < sigmoid).astype(int)

        return np.where(self.global_best == 1)[0] if self.global_best is not None else []