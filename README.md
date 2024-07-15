# Machine Learning Course Projects

This repository contains the collection of projects and assignments completed during the Machine Learning course at KNTU University. Each project folder contains the Jupyter notebooks, reports, and any additional scripts used for the analyses.

## Course Information

- **Instructor:** [Dr.Mahdi Aliyari-Shoorehdeli]
- **TA Lead:** [MJ Ahmadi](https://github.com/MJAHMADEE)
- [Course Website](https://apac.ee.kntu.ac.ir/academic/courses/)
- [Course Github](https://github.com/MJAHMADEE/MachineLearning2024W)

## Projects Overview

Each project in this repository demonstrates the application of various machine learning techniques and algorithms studied throughout the course. The projects are as follows:

1. [Mini Project 1](#mini-project-1)
2. [Mini Project 2](#mini-project-2)
3. [Mini Project 3](#mini-project-3)
4. [Mini Project 4](#mini-project-4)


**Mini-Project 1** involves an in-depth analysis and visualization of datasets. The project includes data exploration, statistical analysis, and various visualizations such as histograms, boxplots, and correlation heatmaps to understand the relationships and distributions of features.

**Mini-Project 2** focuses on detecting faults in bearing data. This project involves loading and preprocessing the dataset, extracting statistical features, normalizing the data, and applying machine learning models such as Multi-Layer Perceptron (MLP) neural networks to classify normal and faulty bearings. It also includes hyperparameter tuning, training different models, and evaluating their performance through accuracy, confusion matrices, and classification reports.

**Mini-Project 3** aims to detect fraudulent transactions using the credit card dataset. The project includes data preprocessing, balancing the dataset using SMOTE, and adding Gaussian noise. An autoencoder is trained to denoise the data, followed by training a neural network classifier to detect fraud. The project also explores various SMOTE sampling strategies and thresholds to optimize recall and accuracy, and visualizes the results through confusion matrices and performance metrics.

**Mini-Project 4** trains Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) agents to solve the Lunar Lander environment. The project involves setting up the environment, defining the DQN and DDQN architectures, implementing experience replay mechanisms, and training the agents with different batch sizes. The performance of the agents is evaluated through rewards and learning curves, and videos of the agent's performance are recorded for visualization.

## Installation and Usage

To run the projects on your local machine, you will need to install the necessary Python packages. It is recommended to use a virtual environment:

```bash
git clone https://github.com/DanialNejad/ML-2024.git
cd ML-2024

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt
```
## Dependencies

The following libraries are required to run the projects:

- Python 3.8 or higher
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook or JupyterLab

All the dependencies are listed in the `requirements.txt` file.

## Contributing

This repository is primarily for educational purposes and personal reference. If you would like to contribute to the projects, your suggestions for improvement or corrections are welcome. Please feel free to submit a pull request or open an issue.

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Should you have any questions, comments, or suggestions, please contact [Danial Nejad](danialabdollahinejad@gmail.com).
- [![LinkedIn][1.1]][1] [LinkedIn Profile](https://www.linkedin.com/in/danial-abdollahi-nejad-883614156)


<!-- Icons -->

[1.1]: https://i.stack.imgur.com/gVE0j.png (linkedin icon without padding)


<!-- Links to your social media accounts -->

[1]: https://www.linkedin.com/in/danial-abdollahi-nejad-883614156
