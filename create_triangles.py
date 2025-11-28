"""Create triangle PNG images for Mesa visualization."""
from PIL import Image, ImageDraw
import os

# Create images folder
os.makedirs("images", exist_ok=True)

def create_top_left_triangle(color, filename, size=50):
    """Create top-left triangle ◤ (fills top-left half of square)."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Triangle: top-left, top-right, bottom-left corners
    points = [(0, 0), (size-1, 0), (0, size-1)]
    draw.polygon(points, fill=color, outline='black')
    
    img.save(filename)
    print(f"Created: {filename}")

def create_bottom_right_triangle(color, filename, size=50):
    """Create bottom-right triangle ◢ (fills bottom-right half of square)."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Triangle: top-right, bottom-right, bottom-left corners
    points = [(size-1, 0), (size-1, size-1), (0, size-1)]
    draw.polygon(points, fill=color, outline='black')
    
    img.save(filename)
    print(f"Created: {filename}")

def create_circle(color, filename, size=50):
    """Create circle for helping pairs."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    margin = 3
    draw.ellipse([margin, margin, size-margin-1, size-margin-1], fill=color, outline='black')
    
    img.save(filename)
    print(f"Created: {filename}")

# Create all images
print("Creating triangle images...")

# Yellow triangles (altruistic)
create_top_left_triangle("#FFE500", "images/yellow_tl.png")
create_bottom_right_triangle("#FFE500", "images/yellow_br.png")

# Blue triangles (non-altruistic)
create_top_left_triangle("#4169E1", "images/blue_tl.png")
create_bottom_right_triangle("#4169E1", "images/blue_br.png")

# Green circle (helping pair)
create_circle("#2ECC71", "images/green_circle.png")

print("\nDone! Images created in 'images/' folder.")

