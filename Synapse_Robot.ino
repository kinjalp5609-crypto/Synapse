//  Synapse_Robot.ino
//  This file lives inside the Arduino on the robot
//  What this does:
//  Waits for commands from laptop via Bluetooth
//  F = Forward, B = Backward, L = Left, R = Right, S = Stop
//  1 = Slow, 2 = Medium, 3 = Fast
//  T = Self test

// LEFT MOTOR pins
#define IN1 2
#define IN2 3
#define ENA 5

// RIGHT MOTOR pins
#define IN3 4
#define IN4 7
#define ENB 6

// Speed settings
#define SPEED_SLOW   100
#define SPEED_MEDIUM 170
#define SPEED_FAST   230
#define SPEED_TURN   160

int currentSpeed = SPEED_MEDIUM;

// If motor spins wrong way then change false to true for that motor
bool LEFT_REVERSED  = false;
bool RIGHT_REVERSED = false;

// Safety: robot stops if no command for 2 seconds
unsigned long lastCommandTime = 0;
#define TIMEOUT_MS 2000

//Setup
void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  stopMotors();
  Serial.begin(9600);

  // Blink 3 times
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_BUILTIN, HIGH); delay(200);
    digitalWrite(LED_BUILTIN, LOW);  delay(200);
  }

  Serial.println("SYNAPSE ROBOT READY");
  lastCommandTime = millis();
}

//Main loop
void loop() {
  // Auto stop if no command for 2 seconds
  if (millis() - lastCommandTime > TIMEOUT_MS) {
    stopMotors();
  }

  if (Serial.available() > 0) {
    char cmd = Serial.read();
    lastCommandTime = millis();
    executeCommand(cmd);
  }
}

//Execute command
void executeCommand(char cmd) {
  switch (cmd) {
    case 'F': case 'f': moveForward();  Serial.println("FORWARD");  break;
    case 'B': case 'b': moveBackward(); Serial.println("BACKWARD"); break;
    case 'L': case 'l': turnLeft();     Serial.println("LEFT");     break;
    case 'R': case 'r': turnRight();    Serial.println("RIGHT");    break;
    case 'S': case 's': stopMotors();   Serial.println("STOP");     break;
    case '1': currentSpeed = SPEED_SLOW;   Serial.println("SLOW");   break;
    case '2': currentSpeed = SPEED_MEDIUM; Serial.println("MEDIUM"); break;
    case '3': currentSpeed = SPEED_FAST;   Serial.println("FAST");   break;
    case 'T': case 't': runSelfTest(); break;
  }
}

//Motor control
void setLeftMotor(int speed, bool forward) {
  if (LEFT_REVERSED) forward = !forward;
  analogWrite(ENA, speed);
  digitalWrite(IN1, forward ? HIGH : LOW);
  digitalWrite(IN2, forward ? LOW  : HIGH);
}

void setRightMotor(int speed, bool forward) {
  if (RIGHT_REVERSED) forward = !forward;
  analogWrite(ENB, speed);
  digitalWrite(IN3, forward ? HIGH : LOW);
  digitalWrite(IN4, forward ? LOW  : HIGH);
}

void moveForward() {
  setLeftMotor(currentSpeed, true);
  setRightMotor(currentSpeed, true);
}

void moveBackward() {
  setLeftMotor(currentSpeed, false);
  setRightMotor(currentSpeed, false);
}

void turnLeft() {
  setLeftMotor(SPEED_TURN, false);
  setRightMotor(SPEED_TURN, true);
}

void turnRight() {
  setLeftMotor(SPEED_TURN, true);
  setRightMotor(SPEED_TURN, false);
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

//Self test(send T to run)
// Run this first to check if motors are correct
void runSelfTest() {
  Serial.println("SELF TEST STARTING...");
  delay(1000);

  Serial.println("Testing LEFT motor forward...");
  setLeftMotor(SPEED_SLOW, true);
  setRightMotor(0, true);
  delay(1500);
  stopMotors();
  delay(500);

  Serial.println("Testing RIGHT motor forward...");
  setLeftMotor(0, true);
  setRightMotor(SPEED_SLOW, true);
  delay(1500);
  stopMotors();
  delay(500);

  Serial.println("Testing FULL FORWARD...");
  moveForward();
  delay(2000);
  stopMotors();
  delay(500);

  Serial.println("Testing TURN LEFT...");
  turnLeft();
  delay(1500);
  stopMotors();
  delay(500);

  Serial.println("Testing TURN RIGHT...");
  turnRight();
  delay(1500);
  stopMotors();

  Serial.println("SELF TEST DONE!");
  Serial.println("If any motor went wrong direction:");
  Serial.println("Set LEFT_REVERSED=true for left motor");
  Serial.println("Set RIGHT_REVERSED=true for right motor");
}
