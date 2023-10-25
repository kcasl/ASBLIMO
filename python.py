import numpy as np
from scipy.signal import lti, impulse
import matplotlib.pyplot as plt

#시스템이 전달하는 함수
numerator = [1]  #분자 다항식
denominator = [1,1,1]  #분모 다항식

#LTI 시스템
lti_system = lti(numerator, denominator)

#디랙 델타 함수 생성 및 시스템에 전달
t, y = impulse(lti_system)

plt.plot(t, y)

plt.xlabel('Time')
plt.ylabel('Impulse Response')
plt.title('Impulse Response of the LTI System')
plt.grid()
plt.show()