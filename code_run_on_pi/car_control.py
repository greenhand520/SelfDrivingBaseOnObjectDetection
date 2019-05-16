import RPi.GPIO as GPIO
import time
import numpy

# set pin
back_motor1 = 35
back_motor2 = 37
back_motor_en = 40  # 后轮电机使能，可输出pwm调速
camera = 38  # 控制摄像头舵机引脚
steer = 36  # 控制方向转动舵机


class CarDriver(object):

    def __init__(self):
        GPIO.setmode(GPIO.BOARD)  # 设置模式
        GPIO.setwarnings(False)
        # 端口为输出模式
        GPIO.setup(back_motor1, GPIO.OUT)
        GPIO.setup(back_motor2, GPIO.OUT)
        GPIO.setup(back_motor_en, GPIO.OUT, initial=False)
        GPIO.setup(steer, GPIO.OUT, initial=False)
        GPIO.setup(camera, GPIO.OUT, initial=False)
        self.forward_speed = 25
        self.turn_speed = 27
        self.back_motor_pwm = GPIO.PWM(back_motor_en, 100)  # 配置PWM
        self.back_motor_pwm.start(0)  # 开始输出PWM
        self.camera_pwm = GPIO.PWM(camera, 50)  # 50Hz
        self.steer_pwm = GPIO.PWM(steer, 50)
        self.camera_pwm.start(0)
        self.steer_pwm.start(0)

    # 向前走
    def car_move_forward(self):
        GPIO.output(back_motor1, GPIO.LOW)
        GPIO.output(back_motor2, GPIO.HIGH)
        self.back_motor_pwm.ChangeDutyCycle(self.forward_speed)  # 改变PWM占空比，参数为占空比

    # 向后退
    def car_move_backward(self):
        GPIO.output(back_motor1, GPIO.HIGH)
        GPIO.output(back_motor2, GPIO.LOW)
        self.back_motor_pwm.ChangeDutyCycle(self.forward_speed)

    # 左拐
    def car_turn_left(self, angle):
        angle = self.return_valid_angle(angle, 30)
        self.control_sg90(angle + 90, self.steer_pwm)
        GPIO.output(back_motor1, GPIO.LOW)
        GPIO.output(back_motor2, GPIO.HIGH)
        self.back_motor_pwm.ChangeDutyCycle(self.forward_speed)

    # 右拐
    def car_turn_right(self, angle):
        print("angle=>", angle)
        angle = self.return_valid_angle(angle, 30)
        self.control_sg90(90 - angle, self.steer_pwm)
        GPIO.output(back_motor1, GPIO.LOW)
        GPIO.output(back_motor2, GPIO.HIGH)
        self.back_motor_pwm.ChangeDutyCycle(self.turn_speed)

    #
    def car_back_left(self, angle):
        angle = self.return_valid_angle(angle, 30)
        self.control_sg90(angle + 90, self.steer_pwm)
        GPIO.output(back_motor1, GPIO.HIGH)
        GPIO.output(back_motor2, GPIO.LOW)
        self.back_motor_pwm.ChangeDutyCycle(self.turn_speed)

    #
    def car_back_right(self, angle):
        print("angle=>", angle)
        angle = self.return_valid_angle(angle, 30)
        self.control_sg90(90 - angle, self.steer_pwm)
        GPIO.output(back_motor1, GPIO.HIGH)
        GPIO.output(back_motor2, GPIO.LOW)
        self.back_motor_pwm.ChangeDutyCycle(self.forward_speed)

    # 清除
    def clean_GPIO(self):
        self.back_motor_pwm.stop()  # 停止输出PWM,后驱电机停止工作
        self.camera_pwm.stop()
        self.steer_pwm.stop()
        GPIO.cleanup()

    # 前轮回正函数
    def car_turn_straight(self):
        self.control_sg90(90, self.steer_pwm)
        time.sleep(0.06)

    # 停止
    def car_stop(self):
        GPIO.output(back_motor1, GPIO.LOW)
        GPIO.output(back_motor2, GPIO.LOW)
        self.control_sg90(180, self.camera_pwm)
        self.car_turn_straight()
        # GPIO.output(back_motor_en, GPIO.LOW) 会导致释放按键无法停止电机转动
        # GPIO.output(front_motor_en, GPIO.LOW)

    def control_by_cmd(self, cmd):

        motor, dire, angle = '0', '0', '0'
        try:
            motor, dire, angle = cmd.split("/")
        except Exception as e:
            print(e)
            motor, dire, angle = '0', '0', '0'
        angle = eval(angle)
        print(motor, dire, angle)
        if motor == '0':
            self.car_stop()
        elif motor == '1':
            print("forward")
            self.car_move_forward()
            self.turn_car(dire, angle)
        elif motor == '2':
            print("backward")
            self.car_move_backward()
            self.turn_car(dire, angle)
        else:
            self.car_stop()

    def turn_car(self, dire, angle):
        if dire == '0':
            self.car_turn_straight()
        elif dire == '1':
            self.car_turn_left(angle)
        else:
            self.car_turn_right(angle)

    def control_sg90(self, degree, sg90_pwm):
        """
        0 ~ 45°之间
        :param degree: angle the sg90 will turn
        :param sg90_pwm: the sg90 to control
        :return: None
        """
        # sg90_pwm.start(0)
        if degree < 0:
            degree = 0
        rate = round(2.5 + degree / 18, 1)
        # print(rate)
        # for i in numpy.arange(2.5, 5, 0.1):
        #     sg90_pwm.ChangeDutyCycle(i)
        #     time.sleep(0.6)
        sg90_pwm.ChangeDutyCycle(rate)
        time.sleep(0.08)  # if the sleepy time is short, sg90 can't reach goal angle
        # sg90_pwm.stop(0)

    def return_valid_angle(self, angle, max_angle):
        if angle > max_angle:
            angle = max_angle
        elif angle < 0:
            angle = 0
        return angle


if __name__ == '__main__':
    car_driver = CarDriver()
    car_driver.control_sg90(65, car_driver.camera_pwm)
    time.sleep(2)
    # control_by_cmd("0/0/0")
    # time.sleep(5)
    # control_by_cmd("1/1/20")
    # time.sleep(5)
    car_driver.control_by_cmd("1/2/30")
    time.sleep(1)
    # car_driver.control_by_cmd("1/2/15")
    # time.sleep(1)
    # car_driver.control_by_cmd("1/2/5")
    # time.sleep(1)
    # control_by_cmd("2/1/30")
    # time.sleep(5)
    # control_by_cmd("2/2/0")
    # time.sleep(5)
    # car_turn_right(30)
    # time.sleep(5)
    car_driver.control_by_cmd("1/0/0")
    time.sleep(1)
    car_driver.control_sg90(180, car_driver.camera_pwm)
    # car_turn_left(30)
    # while True:
    #     car_move_forward()
    car_driver.clean_GPIO()
