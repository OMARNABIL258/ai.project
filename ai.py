import os
import cv2
import numpy as np
from zipfile import ZipFile
from tkinter import *
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import pickle

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition - Assignment 3")
        self.root.geometry("900x700")

        self.X = None
        self.y = None
        self.X_nonface = None
        self.y_nonface = None
        self.labels = {}
        self.pca_models = {}
        self.lda_U = None
        self.lda_nonface_U = None
        self.knn_pca = {}
        self.knn_lda = {}
        self.knn_nonface = {}

        self.create_widgets()
        self.log("Ready. Please load training data.")

    def create_widgets(self):
        frame = Frame(self.root)
        frame.pack(padx=10, pady=10)

        btn_load = Button(frame, text="Load Face Dataset (ZIP)", command=self.load_training_zip)
        btn_load.grid(row=0, column=0, sticky="ew", padx=5)

        btn_load_nonface = Button(frame, text="Load Non-Face Dataset (ZIP)", command=self.load_nonface_zip)
        btn_load_nonface.grid(row=0, column=1, sticky="ew", padx=5)

        btn_train = Button(frame, text="Train Models", command=self.train_models)
        btn_train.grid(row=0, column=2, sticky="ew", padx=5)

        btn_test = Button(frame, text="Test Image", command=self.test_image)
        btn_test.grid(row=0, column=3, sticky="ew", padx=5)

        btn_plots = Button(frame, text="Show Accuracy Plots", command=self.show_plots)
        btn_plots.grid(row=0, column=4, sticky="ew", padx=5)

        Label(frame, text="Split Mode:").grid(row=1, column=0, sticky="w", padx=5)
        self.split_combo = ttk.Combobox(frame, values=["50/50", "70/30"], state="readonly")
        self.split_combo.set("50/50")
        self.split_combo.grid(row=1, column=1, sticky="ew", padx=5)

        Label(frame, text="Method:").grid(row=1, column=2, sticky="w", padx=5)
        self.method_combo = ttk.Combobox(frame, values=["PCA_0.8", "PCA_0.85", "PCA_0.9", "PCA_0.95", "LDA"], state="readonly")
        self.method_combo.set("PCA_0.9")
        self.method_combo.grid(row=1, column=3, sticky="ew", padx=5)

        Label(frame, text="K-NN Neighbors:").grid(row=1, column=4, sticky="w", padx=5)
        self.knn_combo = ttk.Combobox(frame, values=["1", "3", "5", "7"], state="readonly")
        self.knn_combo.set("1")
        self.knn_combo.grid(row=1, column=5, sticky="ew", padx=5)

        self.log_box = Text(self.root, width=100, height=15, state=DISABLED)
        self.log_box.pack(padx=10, pady=10)

        self.result_label = Label(self.root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

    def log(self, msg):
        self.log_box.config(state=NORMAL)
        self.log_box.insert(END, msg + "\n")
        self.log_box.see(END)
        self.log_box.config(state=DISABLED)
        print(msg)

    def load_training_zip(self):
        zip_path = filedialog.askopenfilename(filetypes=[("Zip files", "*.zip")])
        if not zip_path:
            self.log("No face dataset selected.")
            return
        self.log(f"Loading face dataset from: {zip_path}")

        temp_dir = "temp_face_data"
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            self.log("Face dataset extracted successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract face zip: {e}")
            self.log(f"Error extracting face zip: {e}")
            return

        X, y = [], []
        label_map = {}
        for subject_id in range(1, 41):
            folder_path = os.path.join(temp_dir, f's{subject_id}')
            if not os.path.isdir(folder_path):
                self.log(f"Folder s{subject_id} not found.")
                continue
            label_map[subject_id] = f's{subject_id}'
            for i in range(1, 11):
                img_path = os.path.join(folder_path, f'{i}.pgm')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        self.log(f"Failed to read image: {img_path}")
                        continue
                    img = cv2.resize(img, (56, 46))
                    X.append(img.flatten())
                    y.append(subject_id)
                else:
                    self.log(f"Image {img_path} not found.")
        
        if len(X) == 0:
            messagebox.showerror("Error", "No face images found.")
            self.log("No face images loaded.")
            return

        self.X = np.array(X)
        self.y = np.array(y)
        self.labels = label_map
        self.log(f"Loaded {len(self.X)} face images with {len(set(self.y))} classes.")
        messagebox.showinfo("Success", f"Loaded {len(self.X)} face images from {len(set(self.y))} classes.")

    def load_nonface_zip(self):
        zip_path = filedialog.askopenfilename(filetypes=[("Zip files", "*.zip")])
        if not zip_path:
            self.log("No non-face dataset selected.")
            return
        self.log(f"Loading non-face dataset from: {zip_path}")

        temp_dir = "temp_nonface_data"
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            self.log("Non-face dataset extracted successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract non-face zip: {e}")
            self.log(f"Error extracting non-face zip: {e}")
            return

        X_nonface, y_nonface = [], []
        for filename in os.listdir(temp_dir):
            img_path = os.path.join(temp_dir, filename)
            if filename.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg')):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.log(f"Failed to read non-face image: {img_path}")
                    continue
                img = cv2.resize(img, (56, 46))
                X_nonface.append(img.flatten())
                y_nonface.append(0)
            else:
                self.log(f"Skipped unsupported file: {filename}")

        if len(X_nonface) == 0:
            messagebox.showerror("Error", "No non-face images found.")
            self.log("No non-face images loaded.")
            return

        self.X_nonface = np.array(X_nonface)
        self.y_nonface = np.array(y_nonface)
        self.labels[0] = "non-face"
        self.log(f"Loaded {len(self.X_nonface)} non-face images.")
        messagebox.showinfo("Success", f"Loaded {len(self.X_nonface)} non-face images.")

    def split_data(self, X, y, split_mode='50/50'):
        X_train, X_test, y_train, y_test = [], [], [], []
        for subject_id in range(1, 41):
            indices = np.where(y == subject_id)[0]
            if len(indices) != 10:
                self.log(f"Warning: Subject s{subject_id} has {len(indices)} images, expected 10.")
                continue
            if split_mode == '50/50':
                train_indices = indices[::2]
                test_indices = indices[1::2]
            else:  # 70/30
                shuffled = list(indices)
                np.random.shuffle(shuffled)
                train_indices = shuffled[:7]
                test_indices = shuffled[7:]
            X_train.extend(X[train_indices])
            y_train.extend(y[train_indices])
            X_test.extend(X[test_indices])
            y_test.extend(y[test_indices])
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def custom_pca(self, D, alpha):
        pca = PCA(n_components=alpha, svd_solver='full')
        pca.fit(D)
        Ur = pca.components_.T
        mu = pca.mean_
        A = pca.transform(D)
        r = pca.n_components_
        return Ur, mu, A, r

    def custom_lda(self, X, y):
        classes = np.unique(y)
        mu = np.mean(X, axis=0)
        Sb = np.zeros((X.shape[1], X.shape[1]))
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for c in classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mu_c = np.mean(X_c, axis=0)
            Sb += n_c * np.outer(mu_c - mu, mu_c - mu)
            Z_c = X_c - mu_c
            Sw += Z_c.T @ Z_c
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)
        except np.linalg.LinAlgError:
            self.log("LDA eigenvalue computation failed, adding regularization.")
            Sw += 1e-6 * np.eye(Sw.shape[0])
            eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        U = eigenvectors[:, :39]
        A = X @ U
        return U, A

    def train_models(self):
        if self.X is None or self.y is None:
            messagebox.showwarning("Warning", "Please load face dataset first.")
            return
        split_mode = self.split_combo.get()
        self.log(f"Training models with {split_mode} split...")

        X_train, X_test, y_train, y_test = self.split_data(self.X, self.y, split_mode)
        self.log(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # PCA Training
        alphas = [0.8, 0.85, 0.9, 0.95]
        k_values = [1, 3, 5, 7]
        self.knn_pca = {alpha: {} for alpha in alphas}
        for alpha in alphas:
            self.log(f"Training PCA with alpha={alpha}...")
            Ur, mu, X_train_pca, n_components = self.custom_pca(X_train, alpha)
            self.pca_models[alpha] = {'Ur': Ur, 'mu': mu, 'n_components': n_components}
            X_test_pca = (X_test - mu) @ Ur
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
                knn.fit(X_train_pca, y_train)
                self.knn_pca[alpha][k] = knn
                y_pred = knn.predict(X_test_pca)
                acc = accuracy_score(y_test, y_pred)
                self.log(f"PCA (alpha={alpha}, k={k}) accuracy: {acc:.3f}")

        # LDA Training
        self.log("Training LDA...")
        self.lda_U, X_train_lda = self.custom_lda(X_train, y_train)
        X_test_lda = X_test @ self.lda_U
        self.knn_lda = {}
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
            knn.fit(X_train_lda, y_train)
            self.knn_lda[k] = knn
            y_pred = knn.predict(X_test_lda)
            acc = accuracy_score(y_test, y_pred)
            self.log(f"LDA (k={k}) accuracy: {acc:.3f}")

        # Face vs Non-Face Training
        if self.X_nonface is not None:
            self.log("Training face vs non-face classifier...")
            X_combined = np.vstack((self.X, self.X_nonface))
            y_combined = np.hstack((self.y, self.y_nonface))
            X_train_c, X_test_c, y_train_c, y_test_c = self.split_data(X_combined, y_combined, split_mode)
            self.lda_nonface_U, X_train_lda_c = self.custom_lda(X_train_c, y_train_c)
            X_test_lda_c = X_test_c @ self.lda_nonface_U
            self.knn_nonface = {}
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
                knn.fit(X_train_lda_c, y_train_c)
                self.knn_nonface[k] = knn
                y_pred = knn.predict(X_test_lda_c)
                acc = accuracy_score(y_test_c, y_pred)
                self.log(f"Face vs Non-Face LDA (k={k}) accuracy: {acc:.3f}")

        messagebox.showinfo("Success", "Models trained successfully.")

    def test_image(self):
        if not self.pca_models or not self.knn_lda:
            messagebox.showwarning("Warning", "Please train models first.")
            return

        img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.pgm *.jpg *.png *.jpeg")])
        if not img_path:
            self.log("No test image selected.")
            return
        self.log(f"Testing image: {img_path}")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", "Failed to read image.")
            return
        img = cv2.resize(img, (56, 46))
        img_flat = img.flatten().reshape(1, -1)

        method = self.method_combo.get()
        k = int(self.knn_combo.get())
        self.log(f"Predicting with {method}, k={k}")

        # Face vs Non-Face Check
        if self.X_nonface is not None:
            img_lda_nonface = img_flat @ self.lda_nonface_U
            knn_nonface = self.knn_nonface[k]
            pred_nonface = knn_nonface.predict(img_lda_nonface)[0]
            proba_nonface = knn_nonface.predict_proba(img_lda_nonface).max()
            self.log(f"Face vs Non-Face prediction: {self.labels[pred_nonface]} (Confidence: {proba_nonface:.2f})")
            if pred_nonface == 0 and proba_nonface >= 0.7:  # Confidence threshold for non-face
                self.log("Non-face detected with high confidence.")
                self.result_label.config(text="Non-Face Detected", fg="red")
                messagebox.showinfo("Result", "Non-face detected.")
                self.show_matches(img, [], [], is_nonface=True)
                return

        # Face Recognition
        if method.startswith("PCA"):
            alpha = float(method.split('_')[1])
            Ur = self.pca_models[alpha]['Ur']
            mu = self.pca_models[alpha]['mu']
            img_pca = (img_flat - mu) @ Ur
            knn = self.knn_pca[alpha][k]
            pred = knn.predict(img_pca)[0]
            proba = knn.predict_proba(img_pca).max()
        else:  # LDA
            img_lda = img_flat @ self.lda_U
            knn = self.knn_lda[k]
            pred = knn.predict(img_lda)[0]
            proba = knn.predict_proba(img_lda).max()

        # Check if the image is close to any training face
        indices = np.where(self.y == pred)[0]
        dists = [(np.linalg.norm(self.X[idx] - img_flat), idx) for idx in indices]
        if dists:
            min_dist = min(dists, key=lambda x: x[0])[0]
            dist_threshold = 5000  # Adjust based on dataset (empirical)
            if min_dist > dist_threshold:
                self.log(f"Image too far from known faces (distance: {min_dist:.1f}). Classifying as non-face.")
                self.result_label.config(text="Non-Face Detected (Out of Distribution)", fg="red")
                messagebox.showinfo("Result", "Non-face detected (image does not match known faces).")
                self.show_matches(img, [], [], is_nonface=True)
                return

        # Proceed with face recognition
        self.log(f"Prediction: {self.labels[pred]} (Confidence: {proba:.2f})")
        self.result_label.config(text=f"Predicted: {self.labels[pred]} (Confidence: {proba:.2f})", fg="green")
        messagebox.showinfo("Result", f"Predicted: {self.labels[pred]}\nConfidence: {proba:.2f}")

        # Show matches
        dists.sort(key=lambda x: x[0])
        top_matches = dists[:5]
        self.show_matches(img, top_matches, [self.X[idx].reshape(46, 56) for _, idx in top_matches])

    def show_matches(self, test_img, matches, match_images, is_nonface=False):
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 6, 1)
        plt.title("Test Image")
        plt.imshow(test_img, cmap='gray')
        plt.axis('off')

        if not is_nonface:
            for i, (dist, _) in enumerate(matches, start=2):
                plt.subplot(1, 6, i)
                plt.title(f"Match {i-1}\nDist: {dist:.1f}")
                plt.imshow(match_images[i-2], cmap='gray')
                plt.axis('off')
        plt.tight_layout()
        plt.savefig('matches.png')
        plt.show()

    def show_plots(self):
        if not self.pca_models or not self.knn_lda:
            messagebox.showwarning("Warning", "Please train models first.")
            return

        split_mode = self.split_combo.get()
        X_train, X_test, y_train, y_test = self.split_data(self.X, self.y, split_mode)
        alphas = [0.8, 0.85, 0.9, 0.95]
        k_values = [1, 3, 5, 7]
        pca_accs = {k: [] for k in k_values}
        lda_accs = []

        # PCA Accuracies
        for alpha in alphas:
            Ur = self.pca_models[alpha]['Ur']
            mu = self.pca_models[alpha]['mu']
            X_test_pca = (X_test - mu) @ Ur
            for k in k_values:
                knn = self.knn_pca[alpha][k]
                y_pred = knn.predict(X_test_pca)
                acc = accuracy_score(y_test, y_pred)
                pca_accs[k].append(acc)

        # LDA Accuracies
        X_test_lda = X_test @ self.lda_U
        for k in k_values:
            knn = self.knn_lda[k]
            y_pred = knn.predict(X_test_lda)
            acc = accuracy_score(y_test, y_pred)
            lda_accs.append(acc)

        # Plot PCA and LDA accuracies
        plt.figure(figsize=(10, 6))
        for k in k_values:
            plt.plot(alphas, pca_accs[k], marker='o', label=f'PCA (k={k})')
            plt.axhline(y=lda_accs[k_values.index(k)], linestyle='--', label=f'LDA (k={k})')
        plt.title(f"Accuracy vs PCA Explained Variance Ratio ({split_mode} Split)")
        plt.xlabel("PCA Explained Variance Ratio")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig('accuracy_vs_alpha.png')
        plt.show()

        # Non-Face Accuracy Plot
        if self.X_nonface is not None:
            nonface_counts = [50, 100, 200, min(len(self.X_nonface), 400)]
            accuracies = []
            for count in nonface_counts:
                indices = np.random.choice(len(self.X_nonface), count, replace=False)
                X_combined = np.vstack((self.X, self.X_nonface[indices]))
                y_combined = np.hstack((self.y, self.y_nonface[indices]))
                X_train_c, X_test_c, y_train_c, y_test_c = self.split_data(X_combined, y_combined, split_mode)
                U, X_train_lda_c = self.custom_lda(X_train_c, y_train_c)
                X_test_lda_c = X_test_c @ U
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(X_train_lda_c, y_train_c)
                y_pred = knn.predict(X_test_lda_c)
                acc = accuracy_score(y_test_c, y_pred)
                accuracies.append(acc)
                self.log(f"Face vs Non-Face accuracy with {count} non-faces: {acc:.3f}")

            plt.figure(figsize=(10, 6))
            plt.plot(nonface_counts, accuracies, marker='o')
            plt.title("Face vs Non-Face Accuracy vs Number of Non-Face Images")
            plt.xlabel("Number of Non-Face Images")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.savefig('nonface_accuracy.png')
            plt.show()

            self.log("Critique: Accuracy can be misleading with many non-face images due to class imbalance. "
                     "The classifier may favor non-faces, inflating accuracy but reducing face detection sensitivity. "
                     "Metrics like precision, recall, or F1-score are better for imbalanced datasets.")

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()