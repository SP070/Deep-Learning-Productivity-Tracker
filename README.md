# AI-Powered Productivity Tracker

## Overview

This project implements an AI-powered productivity tracker using computer vision and deep learning techniques. It captures video from a webcam, detects faces, estimates gaze and focus, and provides real-time feedback on productivity levels. The system also records the session for later review and analysis.

## Features

- Real-time face detection
- Gaze estimation and focus scoring
- Distraction detection (e.g., prolonged talking)
- Live productivity chart
- Video recording of tracking sessions
- Compatibility with both personal computers and multi-person office environments

## Requirements

- Python 3.9+
- OpenCV
- PyTorch
- torchvision
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/productivity-tracker.git
   cd productivity-tracker
   ```

2. Install the required packages:
   ```
   pip install opencv-python torch torchvision numpy matplotlib
   ```

## Usage

Run the main script:

```
python productivity_tracker.py
```

The script will access your webcam, start tracking productivity, and save a video file of the session.

## How It Works

1. **Face Detection**: The system uses OpenCV's Haar Cascade classifier to detect faces in each frame.
2. **Gaze Estimation**: A pre-trained ResNet18 model estimates the gaze direction and focus level for each detected face.
3. **Productivity Scoring**: Based on the gaze and focus estimates, each person is assigned a productivity score.
4. **Distraction Detection**: The system can detect potential distractions, such as prolonged talking.
5. **Visualization**: A live chart shows productivity trends over time, and face bounding boxes are color-coded (green for focused, red for distracted).
6. **Recording**: The entire session, including all visualizations, is recorded for later review.

7. ![Screenshot 2024-09-05 at 3 20 34 PM](https://github.com/user-attachments/assets/18210600-45a7-4c5d-9702-7254fa5a768d)


## Results

In our testing, the Productivity Tracker successfully:
- Detected faces in various lighting conditions and angles
- Estimated gaze direction with reasonable accuracy
- Provided real-time productivity scores
- Generated clear, informative video recordings of tracking sessions
![Screenshot 2024-09-05 at 3 20 34 PM](https://github.com/user-attachments/assets/eaee7114-0148-4649-907a-6dee13697a83)

The system demonstrated robust performance in both single-user and multi-person scenarios, making it suitable for various work environments.

## Applications in Corporate Settings

This Productivity Tracker can be a valuable tool for companies looking to optimize workforce efficiency and employee well-being:

1. **Performance Optimization**: Managers can identify patterns in productivity fluctuations and make data-driven decisions to improve workflow.

2. **Ergonomic Improvements**: By analyzing gaze patterns, companies can optimize workstation setups to reduce eye strain and improve comfort.

3. **Meeting Efficiency**: The tool can be used to assess engagement levels during meetings, helping to improve meeting structures and durations.

4. **Remote Work Monitoring**: For distributed teams, this provides a non-intrusive way to ensure remote employees are engaged and productive.

5. **Training and Development**: The system can help identify employees who might benefit from additional focus training or support.

6. **Work-Life Balance**: By tracking productivity patterns, companies can encourage better work-life balance, potentially reducing burnout.

7. **Office Layout Optimization**: Aggregate data can inform decisions about office layout to minimize distractions and maximize focus areas.

8. **Compliance and Security**: In industries with strict attention requirements (e.g., security monitoring), the system can ensure operators maintain necessary focus levels.

## Ethical Considerations

While this tool offers significant benefits, it's crucial to implement it ethically:
- Obtain clear consent from all employees before implementation
- Use the data to support and empower employees, not to penalize them
- Ensure all data collection and storage complies with relevant privacy laws and regulations
- Provide employees with access to their own data and the ability to discuss the results
- Regularly review the system's impact on employee morale and company culture

## Future Improvements

- Integrate machine learning for personalized productivity insights
- Add support for emotion detection to gauge employee satisfaction
- Implement privacy-preserving techniques like federated learning
- Develop a user-friendly dashboard for employees to view their own productivity data

## Contributing

Contributions to improve the Productivity Tracker are welcome! Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
