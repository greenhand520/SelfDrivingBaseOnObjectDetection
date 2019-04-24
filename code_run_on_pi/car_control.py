import RPi.GPIO as GPIO
import time
import numpy

# 设定速度，满电时速度太快，图像处理速度跟不上
# 直行快一点，转向慢一点
speed1 = 40 # 直行速度
speed2 = 40  # 拐弯速度

# 后驱电机引脚
back_motor1 = 35
back_motor2 = 37

back_motor_en = 40  # 后轮电机使能，可输出pwm调速

camera = 38 # 控制摄像头舵机引脚
steer = 36 # 控制方向转动舵机

GPIO.setmode(GPIO.BOARD)  # 设置模式
GPIO.setwarnings(False)
# 端口为输出模式
GPIO.setup(back_motor1, GPIO.OUT)
GPIO.setup(back_motor2, GPIO.OUT)
# GPIO.setup(front_motor1, GPIO.OUT)
# GPIO.setup(front_motor2, GPIO.OUT)
GPIO.setup(back_motor_en, GPIO.OUT)
GPIO.setup(steer, GPIO.OUT, initial=False)
GPIO.setup(camera, GPIO.OUT, initial=False)
# 将控制小车运动封装为函数
back_motor_pwm = GPIO.PWM(back_motor_en, 100)  # 配置PWM
back_motor_pwm.start(0)  # 开始输出PWM
camera_pwm = GPIO.PWM(camera, 50) # 50Hz
steer_pwm = GPIO.PWM(steer, 50)


# 向前走
def car_move_forward():
    GPIO.output(back_motor1, GPIO.LOW)
    GPIO.output(back_motor2, GPIO.HIGH)
    back_motor_pwm.ChangeDutyCycle(speed1)  # 改变PWM占空比，参数为占空比


# 向后退
def car_move_backward():
    GPIO.output(back_motor1, GPIO.HIGH)
    GPIO.output(back_motor2, GPIO.LOW)
    back_motor_pwm.ChangeDutyCycle(speed2)


# 左拐
def car_turn_left(angle):
    angle = return_valid_angle(angle, 30)
    control_sg90(angle + 90, steer_pwm)
    GPIO.output(back_motor1, GPIO.LOW)
    GPIO.output(back_motor2, GPIO.HIGH)
    back_motor_pwm.ChangeDutyCycle(speed2)


# 右拐
def car_turn_right(angle):
    angle = return_valid_angle(angle, 30)
    control_sg90(90 - angle, steer_pwm)
    GPIO.output(back_motor1, GPIO.LOW)
    GPIO.output(back_motor2, GPIO.HIGH)
    back_motor_pwm.ChangeDutyCycle(speed2)

#
def car_back_left(angle):
    angle = return_valid_angle(angle, 30)
    control_sg90(angle + 90, steer_pwm)
    GPIO.output(back_motor1, GPIO.HIGH)
    GPIO.output(back_motor2, GPIO.LOW)
    back_motor_pwm.ChangeDutyCycle(speed2)

#
def car_back_right(angle):
    angle = return_valid_angle(angle, 30)
    control_sg90(90 - angle, steer_pwm)
    GPIO.output(back_motor1, GPIO.HIGH)
    GPIO.output(back_motor2, GPIO.LOW)
    back_motor_pwm.ChangeDutyCycle(speed2)


# 清除
def clean_GPIO():
    back_motor_pwm.stop()  # 停止输出PWM,后驱电机停止工作
    camera_pwm.stop()
    steer_pwm.stop()
    GPIO.cleanup()



# 前轮回正函数
def car_turn_straight():
    control_sg90(90, steer_pwm)
    time.sleep(0.06)


# 停止
def car_stop():
    GPIO.output(back_motor1, GPIO.LOW)
    GPIO.output(back_motor2, GPIO.LOW)
    car_turn_straight()
    # GPIO.output(back_motor_en, GPIO.LOW) 会导致释放按键无法停止电机转动
    # GPIO.output(front_motor_en, GPIO.LOW)

def control_by_cmd(dire, sleep_time, angle):
    print(dire, angle)
    if dire == '2':
        car_turn_straight()
        car_move_forward()
    elif dire == '3':
        car_turn_straight()
        car_move_backward()
    elif dire == '0':
        car_turn_left(angle)
        time.sleep(sleep_time)
    elif dire == '1':
        car_turn_right(angle)
        time.sleep(sleep_time)
    else:
        car_stop()

def control_sg90(degree, sg90_pwm):
    """
    0 ~ 45°之间
    :param degree: angle the sg90 will turn
    :param sg90_pwm: the sg90 to control
    :return: None
    """
    sg90_pwm.start(0)
    if degree < 0:
        degree = 0
    rate = round(2.5 + degree / 18, 1)
    # print(rate)
    # for i in numpy.arange(2.5, 5, 0.1):
    #     sg90_pwm.ChangeDutyCycle(i)
    #     time.sleep(0.02)
    sg90_pwm.ChangeDutyCycle(rate)
    time.sleep(0.6)    # if the sleepy time is short, sg90 can't reach goal angle
    sg90_pwm.stop()    # solve sg90 shake

def return_valid_angle(angle, max_angle):
    if angle > max_angle:
        angle = max_angle
    elif angle < 0:
        angle = 0
    return angle

if __name__ == '__main__':

    # car_turn_straight()
    # car_move_forward()
    # while True:
    #     car_turn_left()
    # time.sleep(10)
    control_sg90(110, steer_pwm)
    clean_GPIO()
