# 🐾 Animal Detection Model using YOLOv8 and Streamlit

This project detects and classifies **animals in images and videos** using **YOLOv8 (Ultralytics)** and provides a clean **Streamlit GUI** interface.

It highlights:
- 🟥 **Carnivorous animals** (like lions, tigers, and wolves)  
- 🟩 **Non-carnivorous animals** (like elephants, cows, horses, etc.)  
- Displays a **warning pop-up** with the number of carnivorous animals detected  

---

## 🧠 Project Overview
This project was developed as part of my **internship**.  
The goal was to **train an Animal Detection Model** that can identify multiple species in a single image or video frame, highlight carnivores in red, and provide a user-friendly GUI.

I have also **trained my own custom YOLO model** using the available datasets.  
However, I want to mention **humbly** that:
> Some animal classes (especially wild ones like *lion, tiger, cheetah*, etc.) had **limited or insufficient data**,  
> so the trained model may not perfectly detect all such species in every scenario.

Despite this, the model performs strongly on common animals and general categories — it demonstrates the full workflow of **model training, testing, and real-time GUI integration** successfully.

---

## 🚀 Features
✅ Detects multiple animals in a single image or video frame  
✅ Highlights **carnivores in red** and **others in green**  
✅ Pop-up warning showing total carnivores detected  
✅ Clean, **no-sidebar Streamlit interface**  
✅ Works for both **images** and **videos**  
✅ Option to **download the annotated result**

---

## 🧰 Tech Stack
| Component | Technology |
|------------|-------------|
| Framework | [Streamlit](https://streamlit.io/) |
| Model | [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) |
| Language | Python 3.9+ |
| Libraries | OpenCV, NumPy, Pillow, Torch |

---

## 📁 Project Structure

