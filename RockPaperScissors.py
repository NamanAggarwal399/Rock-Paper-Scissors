import cv2
import mediapipe as mp
import random
from collections import deque
import statistics as st


class HandRecognition:
    def __init__(self):
        self.hand_valid = False
        self.hand_number = 0
        self.is_counting = False

    def detect_hand(self, img, cf, ss, ge, hds, mph, mpd, mpds):
        self.hand_number = 0
        hand_landmarks = []
        cf.count = 0
        self.is_counting = False

        img = cv2.flip(img, 1)

        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hds.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if res.multi_hand_landmarks:
            self.is_counting = True

            if ge.player_choice != "Nothing" and not self.hand_valid:
                self.hand_valid = True
                ge.cpu_choice = random.choice(ge.cpu_choices)
                ge.winner, ge.winner_colour = ss.calculate_score(ge.player_choice, ge.cpu_choice)

            for hand in res.multi_hand_landmarks:
                mpd.draw_landmarks(
                    img,
                    hand,
                    mph.HAND_CONNECTIONS,
                    mpds.get_default_hand_landmarks_style(),
                    mpds.get_default_hand_connections_style()
                )

                label = res.multi_handedness[self.hand_number].classification[0].label

                for id, landmark in enumerate(hand.landmark):
                    img_h, img_w, img_channel = img.shape
                    x_pos, y_pos = int(landmark.x * img_w), int(landmark.y * img_h)

                    hand_landmarks.append([id, x_pos, y_pos, label])

                cf.count_fingers(hand_landmarks)
                self.hand_number += 1
        else:
            self.hand_valid = False

        return img


class GameEnvironment:
    def __init__(self):
        self.player_choice = "Nothing"
        self.values = ["Rock", "Invalid", "Scissors", "Invalid", "Invalid", "Paper"]
        self.cpu_choices = ["Rock", "Paper", "Scissors"]
        self.cpu_choice = "Nothing"
        self.winner = "None"
        self.winner_colour = (0, 255, 0)

    def update_player_choice(self, count, is_counting):
        if is_counting and count <= 5:
            self.player_choice = self.values[count]
        elif is_counting:
            self.player_choice = "Invalid"
        else:
            self.player_choice = "Nothing"

    def op(self, img, player_score, cpu_score):
        cv2.putText(img, "You", (90, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(img, "CPU", (350, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.putText(img, self.player_choice, (45, 375),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(img, self.cpu_choice, (305, 375),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.putText(img, self.winner, (200, 450),
                    cv2.FONT_HERSHEY_DUPLEX, 2, self.winner_colour, 5)

        cv2.putText(img, str(player_score), (145, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(img, str(cpu_score), (405, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.imshow('Rock-Paper-Scissors', img)


class CountFingers:
    def __init__(self):
        self.count = 0

    def count_fingers(self, hand_landmarks):
        # Index Finger
        if hand_landmarks[8][2] < hand_landmarks[6][2]:
            self.count += 1

        # Middle Finger
        if hand_landmarks[12][2] < hand_landmarks[10][2]:
            self.count += 1

        # Ring Finger
        if hand_landmarks[16][2] < hand_landmarks[14][2]:
            self.count += 1

        # Pinky Finger
        if hand_landmarks[20][2] < hand_landmarks[18][2]:
            self.count += 1

        # Thumb
        if hand_landmarks[4][3] == "Left" and hand_landmarks[4][1] > hand_landmarks[3][1]:
            self.count += 1
        elif hand_landmarks[4][3] == "Right" and hand_landmarks[4][1] < hand_landmarks[3][1]:
            self.count += 1


class ScoringSystem:
    def __init__(self):
        self.player_score = 0
        self.computer_score = 0

    @staticmethod
    def calculate_winner(cpu_choice, player_choice):
        if player_choice == "Invalid":
            return "Invalid!"

        if player_choice == cpu_choice:
            return "Tie!"

        elif player_choice == "Rock" and cpu_choice == "Scissors":
            return "You win!"

        elif player_choice == "Rock" and cpu_choice == "Paper":
            return "CPU wins!"

        elif player_choice == "Scissors" and cpu_choice == "Rock":
            return "CPU wins!"

        elif player_choice == "Scissors" and cpu_choice == "Paper":
            return "You win!"

        elif player_choice == "Paper" and cpu_choice == "Rock":
            return "You win!"

        elif player_choice == "Paper" and cpu_choice == "Scissors":
            return "CPU wins!"

    def calculate_score(self, player_choice, cpu_choice):
        wnr = self.calculate_winner(cpu_choice, player_choice)

        if wnr == "You win!":
            self.player_score += 1
            wnr_clr = (0, 255, 0)
        elif wnr == "CPU wins!":
            self.computer_score += 1
            wnr_clr = (0, 0, 255)
        else:
            wnr_clr = (255, 0, 0)

        return wnr, wnr_clr


# Game Implementation
hand_recognition = HandRecognition()
game_environment = GameEnvironment()
count_fingers = CountFingers()
scoring_system = ScoringSystem()

webcam = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
de = deque(['Nothing'] * 5, maxlen=5)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.4
) as hands:
    while webcam.isOpened():
        success, image = webcam.read()
        if not success:
            print("Camera isn't working")
            continue

        image = hand_recognition.detect_hand(image, count_fingers, scoring_system, game_environment, hands, mp_hands,
                                             mp_drawing, mp_drawing_styles)
        game_environment.update_player_choice(count_fingers.count, hand_recognition.is_counting)
        de.appendleft(game_environment.player_choice)

        try:
            game_environment.player_choice = st.mode(de)
        except st.StatisticsError:
            print("Stats Error")
            continue

        game_environment.op(image, scoring_system.player_score, scoring_system.computer_score)

        if cv2.waitKey(1) & 0xFF == 27:
            break

webcam.release()
cv2.destroyAllWindows()
