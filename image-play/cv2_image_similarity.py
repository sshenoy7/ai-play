import cv2

print(dir(cv2))

def compare_images(image1_path, image2_path):
  """
  Compares two images using SSIM and returns the similarity score.
  """
  image1 = cv2.imread(image1_path)
  image2 = cv2.imread(image2_path)

  # Resize images to the same size (assuming image1 is the reference)
  image2 = cv2.resize(image2, dsize=(image1.shape[1], image1.shape[0]))

  psnr_score = cv2.PSNR(image1, image2)
  return psnr_score

# Example usage
image1_path = "V.png"
image2_path = "ZC1.png"
similarity_score = compare_images(image1_path, image2_path)
print(f"Similarity Score: {similarity_score}")

if __name__ == "__main__":
  compare_images(image1_path, image2_path)
