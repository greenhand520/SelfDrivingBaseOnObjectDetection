import RPi.GPIO as GPIO
import time

# 设定速度，满电时速度太快，图像处理速度跟不上
# 直行快一点，转向慢一点
speed1 = 40 # 直行速度
speed2 = 50  # 拐弯速度
# 后驱电机引脚
back_motor1 = 36
back_motor2 = 38
# 前驱电机引脚
front_motor1 = 35
front_motor2 = 37

back_motor_en = 40  # 后轮电机使能，可输出pwm调速
front_motor_en = 33  # 前轮电机是能, 直接输出高低电平控制电机是否工作

GPIO.setmode(GPIO.BOARD)  # 设置模式
GPIO.setwarnings(False)
# 端口为输出模式
GPIO.setup(back_motor1, GPIO.OUT)
GPIO.setup(back_motor2, GPIO.OUT)
GPIO.setup(front_motor1, GPIO.OUT)
GPIO.setup(front_motor2, GPIO.OUT)
GPIO.setup(back_motor_en, GPIO.OUT)
GPIO.setup(front_motor_en, GPIO.OUT)
# 将控制小车运动封装为函数
back_motor_pwm = GPIO.PWM(back_motor_en, 100)  # 配置PWM
back_motor_pwm.start(0)  # 开始输出PWM


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
def car_turn_left():
    GPIO.output(front_motor_en, GPIO.HIGH)  # 输入高电平，让前轮转向电机正常工作。
    GPIO.output(front_motor1, GPIO.HIGH)
    GPIO.output(front_motor2, GPIO.LOW)
    GPIO.output(back_motor1, GPIO.LOW)
    GPIO.output(back_motor2, GPIO.HIGH)
    back_motor_pwm.ChangeDutyCycle(speed2)


# 右拐
def car_turn_right():
    GPIO.output(front_motor_en, GPIO.HIGH)
    GPIO.output(front_motor1, GPIO.LOW)
    GPIO.output(front_motor2, GPIO.HIGH)
    GPIO.output(back_motor1, GPIO.LOW)
    GPIO.output(back_motor2, GPIO.HIGH)
    back_motor_pwm.ChangeDutyCycle(speed2)


#
def car_back_left():
    GPIO.output(front_motor_en, GPIO.HIGH)
    GPIO.output(front_motor1, GPIO.HIGH)
    GPIO.output(front_motor2, GPIO.LOW)
    GPIO.output(back_motor1, GPIO.HIGH)
    GPIO.output(back_motor2, GPIO.LOW)
    back_motor_pwm.ChangeDutyCycle(speed2)


#
def car_back_right():
    GPIO.output(front_motor_en, GPIO.HIGH)
    GPIO.output(front_motor1, GPIO.LOW)
    GPIO.output(front_motor2, GPIO.HIGH)
    GPIO.output(back_motor1, GPIO.HIGH)
    GPIO.output(back_motor2, GPIO.LOW)
    back_motor_pwm.ChangeDutyCycle(speed2)


# 清除
def clean_GPIO():
    GPIO.cleanup()
    back_motor_pwm.stop()  # 停止输出PWM,后驱电机停止工作


# 前轮回正函数
def car_turn_straight():
    GPIO.output(front_motor_en, GPIO.LOW)  # 前驱电机不工作，不受力，方向回正
    time.sleep(0.05)


# 停止
def car_stop():
    GPIO.output(back_motor1, GPIO.LOW)
    GPIO.output(back_motor2, GPIO.LOW)
    # GPIO.output(back_motor_en, GPIO.LOW) 会导致释放按键无法停止电机转动
    GPIO.output(front_motor_en, GPIO.LOW)

def control_by_cmd(cmd, sleep_time):
    if cmd == '2':
        car_turn_straight()
        car_move_forward()
    elif cmd == '3':
        car_turn_straight()
        car_move_backward()
    elif cmd == '0':
        car_turn_left()
        time.sleep(sleep_time)
    elif cmd == '1':
        car_turn_right()
        time.sleep(sleep_time)
    else:
        car_stop()


if __name__ == '__main__':
    car_turn_straight()
    # car_move_forward()
    car_move_backward()
    time.sleep(10)
    clean_GPIO()
