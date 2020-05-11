import src.data as dat
import src.networks as net
import src.config as cfg
import tkinter as tk
from tkinter import filedialog


def print_menu(value):
    if value == 0:
        print("\nThis is the main menu. Choose an option:\n1. Missing files or unknown errors about the network\n2. "
              "SMV + HOG\n3. SVM + FAST\n4. Xception\n5. Exit")
    elif value == 1:
        print("\n1. Recreate directories\n2. Recreate data files\n3. Train all networks\n4. Load all networks")
    elif value == 2:
        print("\n1. Train network\n2. Reload network\n3. Predict file")
    elif value == 3:
        print("\n1. Train network\n2. Reload network\n3. Predict file")
    elif value == 4:
        print("\n1. Train network\n2. Reload network\n3. Predict file\n4. Create files based model.h5")
    elif value == 5:
        print("\n")


def init():
    if dat.check_folders():
        print("\nMissing directories and data files.")
        print("Beginning reconstruction.\n")
        dat.create_directories()
        dat.create_data_files()
        print("Training models.\n")
        net.train_svm_hog()
        net.train_svm_fast()
        net.train_xception()
    elif not dat.check_folders():
        print("\nAll directories accounted for.\n")
    print("\nContinuing with the program.\n")


if __name__ == '__main__':
    init()
    root = tk.Tk()
    root.withdraw()
    model_hog = net.load_hog_model()
    model_fast = net.load_fast_model()
    model_x = net.load_xception()
    while True:
        print_menu(0)
        option = input("Enter the option number: ")
        if str(option) == "1":
            print_menu(1)
            inside_option = input("Enter the option number: ")
            if str(inside_option) == "1":
                print("Beginning reconstruction.\n")
                dat.create_directories()
            elif str(inside_option) == "2":
                print("Beginning reconstruction.\n")
                dat.create_data_files()
            elif str(inside_option) == "3":
                print("Training models.\n")
                net.train_svm_hog()
                net.train_svm_fast()
                net.train_xception()
            elif str(inside_option) == "4":
                print("Loading models.\n")
                model_hog = net.load_hog_model()
                model_fast = net.load_fast_model()
                model_x = net.load_xception()
        elif str(option) == "2":
            print_menu(2)
            inside_option = input("Enter the option number: ")
            if str(inside_option) == "1":
                print("\nTraining SVM+HOG network.\n")
                net.train_svm_hog()
            elif str(inside_option) == "2":
                print("\nLoading SVM+HOG network.\n")
                model_hog = net.load_hog_model()
            elif str(inside_option) == "3":
                print("\nPredicting using SVM+HOG network.\n")
                file_path = filedialog.askopenfilename()
                print(net.do_prediction(file_path, "hog", model_hog))
        elif str(option) == "3":
            print_menu(3)
            inside_option = input("Enter the option number: ")
            if str(inside_option) == "1":
                print("\nTraining SVM+FAST network.\n")
                net.train_svm_fast()
            elif str(inside_option) == "2":
                print("\nLoading SVM+FAST network.\n")
                model_fast = net.load_fast_model()
            elif str(inside_option) == "3":
                print("\nPredicting using SVM+FAST network.\n")
                file_path = filedialog.askopenfilename()
                print(net.do_prediction(file_path, "fast", model_fast))
        elif str(option) == "4":
            print_menu(4)
            inside_option = input("Enter the option number: ")
            if str(inside_option) == "1":
                print("\nTraining Xception network.\n")
                net.train_xception()
            elif str(inside_option) == "2":
                print("\nLoading Xception network.\n")
                model_x = net.load_xception()
            elif str(inside_option) == "3":
                print("\nPredicting using Xception network.\n")
                file_path = filedialog.askopenfilename()
                print(net.do_prediction(file_path, "x", model_x))
            elif str(inside_option) == "4":
                print("\nRebuilding some Xception files.\n")
                net.create_x_files()
        elif str(option) == "5":
            print("\nClosing the program.")
            exit()
