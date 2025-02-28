# GAN-Augmented-Multimodal-Biometrics-Enhancing-Security-and-Privacy
A multimodal biometric system integrating voice and vein to enhance authentication security for sensitive places. Implemented advanced feature fusion techniques and applied steganography to embed biometric data securely into digital images. Tested system robustness against synthetic biometric attacks using Generative Adversarial Networks (GANs)

## Key Features  
- **Multimodal Authentication:** Fusion of vein and voice biometric data  
- **Feature Extraction & Fusion:** Extract meaningful patterns from images and audio  
- **Steganography:** Securely embeds biometric data into images  
- **GAN-Based Enhancement:** Uses GANs to improve data generation and security  
- **Security Against Attacks:** Protects against spoofing and synthetic identity fraud  

## Dataset  
- **Finger Vein Dataset:** Contains images of finger vein patterns  
- **Voice Dataset:** Converted into spectrograms for feature extraction  
- **Preprocessing:** Includes resizing, normalization, and feature extraction  

## System Architecture  
1. **Data Collection:** Finger vein images and voice recordings  
2. **Feature Extraction:** Using **LBP/Gabor filters** for vein and **MFCCs** for voice  
3. **Feature Fusion:** Combining extracted features  
4. **Steganographic Embedding:** Hiding fused biometric data in an image  
5. **GAN Training:** Improving biometric feature security and robustness  
6. **GAN Testing:** Validating performance  
7. **Analysis Report:** Evaluating accuracy and security improvements  

## Installation & Requirements  
### **Prerequisites**  
- Python 3.8+  
- TensorFlow / PyTorch  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

### **Setup**  
```bash
git clone https://github.com/yourusername/multimodal-biometrics-gan.git
cd multimodal-biometrics-gan
pip install -r requirements.txt
