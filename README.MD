# MyProject

This document provides instructions on how to install the necessary dependencies, run the application, and monitor AI training using TensorBoard.

## Installation

This project uses a `pyproject.toml` file to manage dependencies. To install them, ensure you have Python and pip installed. Then, navigate to the root directory of the project in your terminal or command prompt and run the following command:

```bash
pip install .

If you are using a specific virtual environment manager like Poetry or PDM, you might need to use their respective commands:

For Poetry:

poetry install


Running the Application
The main application is a Streamlit app. To run it, execute the following command in your terminal or command prompt from the project's root directory:

streamlit run app.py

This will typically open the application in your default web browser.

AI Monitoring with TensorBoard
To monitor AI model training, performance, and other metrics, this project uses TensorBoard. The logs are stored in the models/DPT/logs directory.

To launch TensorBoard, run the following command in your terminal or command prompt:

tensorboard --logdir=models/DPT/logs

After running this command, TensorBoard will provide a URL (usually http://localhost:6006/). Open this URL in your web browser to view the dashboard.

Make sure all paths and commands are correct for your specific project structure.