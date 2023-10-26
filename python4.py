import numpy as np
import matplotlib.pyplot as plt

# 초기 상태 및 초기 공분산
initial_state = 4.5
initial_covariance = 0.1

# 시스템 모델
A = 1  # 시스템 모델의 변환 행렬
Q = 0.1  # 프로세스 노이즈 공분산

# 측정 모델
H = 1  # 측정 모델의 변환 행렬
R = 0.5  # 측정 노이즈 공분산

# 파티클 필터 실행
num_iterations = 50
state_estimates = []

state = initial_state
covariance = initial_covariance

for _ in range(num_iterations):
    # Prediction Step (시스템 모델을 사용하여 예측)
    state = A * state
    covariance = A * covariance * A + Q
    
    # Update Step (측정을 사용하여 업데이트)
    measurement = state + np.random.normal(0, R)
    K = covariance * H / (H * covariance * H + R)
    state = state + K * (measurement - H * state)
    covariance = (1 - K * H) * covariance
    
    state_estimates.append(state)

# 결과 시각화
true_states = [initial_state]
for _ in range(num_iterations - 1):
    true_states.append(A * true_states[-1])

plt.figure()
plt.plot(range(num_iterations), true_states, label="True State")
plt.plot(range(num_iterations), state_estimates, label="Estimated State")
plt.xlabel("Iteration")
plt.ylabel("State")
plt.legend()
plt.show()
