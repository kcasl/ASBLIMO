import numpy as np
import matplotlib.pyplot as plt

num_particles = 1000

#파티클 무작위 생성
particles = np.random.uniform(0, 10, num_particles)

#파티클에 무작위로 "노이즈" 추가
def motion_model(particles, movement_noise=0.1):
    particles += np.random.normal(0, movement_noise, num_particles)
    return particles

#파티클의 가중치 계산
def measurement_model(particles, measurement, measurement_noise=0.5):
    likelihood = np.exp(-0.5 * ((particles - measurement) / measurement_noise)**2)
    weights = likelihood / np.sum(likelihood)
    return weights

#파티클 리샘플링
def resample(particles, weights):
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[indices]
    return particles

true_state = 4.5

#파티클 필터 실행
num_iter = 50
for i in range(num_iter):
    #파티클 이동
    particles = motion_model(particles)
    
    #측정
    measurement = true_state + np.random.normal(0, 0.5)
    weights = measurement_model(particles, measurement)
    particles = resample(particles, weights)
    estimated_state = np.mean(particles)
    
    print(f"Iter {i+1}: True state = {true_state:.2f}, Estimated state = {estimated_state:.2f}")

plt.figure()
plt.hist(particles, bins=30, density=True, alpha=0.5, label="Particle filter")
plt.axvline(true_state, color="r", label="True state")
plt.xlabel("State")
plt.ylabel("Probability")
plt.legend()
plt.show()
