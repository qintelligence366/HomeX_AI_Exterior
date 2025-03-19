from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
import cloudinary
import cloudinary.uploader
import time
import traceback
import numpy as np
import cv2

# 配置 Cloudinary
cloudinary.config(
    cloud_name="djepeyshu",
    api_key="188513436699751",
    api_secret="h9U86BmSDxSpQuYKL_S5tXg6odQ"
)

# 定义请求体模型
class ImageRequest(BaseModel):
    imageUrl: str
    productUrl: str
    regions: list

    class Config:
        arbitrary_types_allowed = True

# 创建 FastAPI 应用
app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件到 /static
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# 判断是否为侧面图
def is_side_view(points, threshold=0.18):
    if len(points) < 4:
        return False
    points_array = np.array(points, dtype=np.float32)
    top_points = sorted(points_array, key=lambda p: p[1])[:2]
    bottom_points = sorted(points_array, key=lambda p: p[1], reverse=True)[:2]
    top_points = sorted(top_points, key=lambda p: p[0])
    bottom_points = sorted(bottom_points, key=lambda p: p[0])
    left_points = sorted(points_array, key=lambda p: p[0])[:2]
    right_points = sorted(points_array, key=lambda p: p[0], reverse=True)[:2]
    left_points = sorted(left_points, key=lambda p: p[1])
    right_points = sorted(right_points, key=lambda p: p[1])

    top_width = np.sqrt((top_points[1][0] - top_points[0][0])**2 + (top_points[1][1] - top_points[0][1])**2)
    bottom_width = np.sqrt((bottom_points[1][0] - bottom_points[0][0])**2 + (bottom_points[1][1] - bottom_points[0][1])**2)
    left_height = np.sqrt((left_points[1][0] - left_points[0][0])**2 + (left_points[1][1] - left_points[0][1])**2)
    right_height = np.sqrt((right_points[1][0] - right_points[0][0])**2 + (right_points[1][1] - right_points[0][1])**2)

    width_diff = abs(top_width - bottom_width) / max(top_width, bottom_width)
    height_diff = abs(left_height - right_height) / max(left_height, right_height)
    angle_top = np.arctan2(top_points[1][1] - top_points[0][1], top_points[1][0] - top_points[0][0])
    angle_bottom = np.arctan2(bottom_points[1][1] - bottom_points[0][1], bottom_points[1][0] - bottom_points[0][0])
    angle_diff = abs(angle_top - angle_bottom)

    print(f"width_diff: {width_diff}, height_diff: {height_diff}, angle_diff (degrees): {np.degrees(angle_diff)}")
    return width_diff > threshold and height_diff > threshold and angle_diff > np.radians(10)

# 透视变换
def apply_perspective_transform(product_resized, pixel_coords, x, y, w, h, customer_image):
    try:
        dst_points = pixel_coords[:4].astype(np.float32)
        src_points = np.array([
            [0, 0],
            [product_resized.shape[1] - 1, 0],
            [product_resized.shape[1] - 1, product_resized.shape[0] - 1],
            [0, product_resized.shape[0] - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        product_warped = cv2.warpPerspective(product_resized, matrix, (customer_image.shape[1], customer_image.shape[0]))
        return product_warped[y:y+h, x:x+w]
    except cv2.error as e:
        print("透视变换错误:", e)
        return product_resized

# 替换车库门函数
@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        print("Received request:", request.dict())
        img_response = requests.get(request.imageUrl.split("?")[0], timeout=10)
        if img_response.status_code != 200:
            raise Exception(f"Failed to download image: Status {img_response.status_code}")
        customer_image = cv2.cvtColor(np.array(Image.open(BytesIO(img_response.content)).convert("RGBA")), cv2.COLOR_RGBA2BGRA)

        if not request.productUrl or not request.productUrl.strip():
            raise Exception("productUrl is empty or invalid")
        prod_response = requests.get(request.productUrl.split("?")[0], timeout=10)
        if prod_response.status_code != 200:
            raise Exception(f"Failed to download product: Status {prod_response.status_code}")
        product_img = cv2.cvtColor(np.array(Image.open(BytesIO(prod_response.content)).convert("RGBA")), cv2.COLOR_RGBA2BGRA)

        regions_data = request.regions
        if not isinstance(regions_data, list) or len(regions_data) < 1 or not isinstance(regions_data[0], list):
            raise Exception("Invalid regions format, need at least one list of points")
        region_points = regions_data[0]
        if len(region_points) < 4:
            raise Exception("Invalid regions format, need at least 4 points")
        original_height, original_width = customer_image.shape[:2]
        pixel_coords = np.array([[int(coord["x"] * original_width), int(coord["y"] * original_height)] for coord in region_points], dtype=np.int32)
        print("Pixel coordinates:", pixel_coords)

        mask = np.zeros((original_height, original_width), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_coords], 255)
        print("Mask created")

        x, y, w, h = cv2.boundingRect(pixel_coords)
        print("Boundary box (x, y, w, h):", (x, y, w, h))

        product_resized = cv2.resize(product_img, (w, h), interpolation=cv2.INTER_AREA)
        print("Resized product image size:", product_resized.shape)

        product_warped = product_resized
        if is_side_view(pixel_coords) and len(pixel_coords) >= 4:
            print("执行透视变换...")
            product_warped = apply_perspective_transform(product_resized, pixel_coords, x, y, w, h, customer_image)
        else:
            print("不执行透视变换，使用默认替换")

        kernel_size = 15
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        print("Edge feathering completed")
        mask_region = blurred_mask[y:y+h, x:x+w]
        print("Mask region shape:", mask_region.shape)

        if product_warped.shape[2] == 4:
            print("Product image has alpha channel")
            alpha = cv2.resize(product_warped[:, :, 3] / 255.0, (w, h), interpolation=cv2.INTER_LINEAR)
            product_rgb = cv2.resize(product_warped[:, :, :3], (w, h), interpolation=cv2.INTER_LINEAR)
            for c in range(3):
                customer_image[y:y+h, x:x+w, c] = np.where(
                    blurred_mask[y:y+h, x:x+w] > 0,
                    (1 - alpha) * customer_image[y:y+h, x:x+w, c] + alpha * product_rgb[:, :, c],
                    customer_image[y:y+h, x:x+w, c]
                )
        else:
            print("No alpha channel, direct overwrite")
            for c in range(3):
                customer_image[y:y+h, x:x+w, c] = np.where(
                    blurred_mask[y:y+h, x:x+w] > 0,
                    product_warped[:, :, c],
                    customer_image[y:y+h, x:x+w, c]
                )

        img_result = Image.fromarray(cv2.cvtColor(customer_image, cv2.COLOR_BGRA2RGB))
        img_byte_arr = BytesIO()
        img_result.save(img_byte_arr, format="JPEG")
        upload_result = cloudinary.uploader.upload(
            img_byte_arr.getvalue(),
            public_id=f"garage_doors/replaced_{int(time.time())}",
            format="jpg"
        )
        new_image_url = upload_result["url"]

        return {"imageUrl": new_image_url}
    except Exception as e:
        print("Error processing image:", traceback.format_exc())
        return {"error": str(e)}, 500
