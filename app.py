import streamlit as st
from ultralytics import YOLO
import PIL
from PIL import ImageEnhance
import setting

# st.set_page_config MUST be the first Streamlit command executed
setting.configure_page()

import cv2
import numpy as np
import io          # F4/F7: needed to re-open image bytes inside cached function
import pathlib     # GN fix: used to save uploaded image to disk for gn_classifier.py
import pandas as pd

import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()
# Support both local .env files and Streamlit Community Cloud secrets
@st.cache_resource
def get_inference_client():
    try:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["ROBOFLOW_API_KEY"]
            except Exception:
                pass
            
        return InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
    except Exception:
        return None

import wall_detector
import zipfile
import json

# F1: load_model is defined here (module level, before main()) so the
# @st.cache_resource decorator persists the 52 MB model across all reruns.
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Load and cache the YOLO model. Called once per session."""
    return YOLO(model_path)


# F7: @st.cache_data caches the expensive preprocessing result keyed on the
# raw image bytes. Preprocessing no longer reruns on slider/label changes.
@st.cache_data(show_spinner="Preprocessing floor plan\u2026")
def preprocess_floorplan(image_bytes: bytes, darken_strength: float = 2.3):
    """
    Grayscale-only preprocessing that darkens floor-plan drawings
    while preserving faint edges, door arcs, windows, dimensions, etc.
    Accepts raw image bytes so Streamlit can hash and cache the result.
    """
    # F7: reconstruct the PIL image from bytes inside the cached function
    pil_image = PIL.Image.open(io.BytesIO(image_bytes))

    # Handle transparent PNGs on white background
    if pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info:
        rgba = pil_image.convert("RGBA")
        white = PIL.Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil_image = PIL.Image.alpha_composite(white, rgba)

    # Force grayscale input
    gray = np.array(pil_image.convert("L"), dtype=np.uint8)

    h, w = gray.shape[:2]
    
    # NEW: Find content bounding box (auto-crop white margins)
    nw = gray < 250
    rows = np.any(nw, axis=1)
    cols = np.any(nw, axis=0)
    if rows.any() and cols.any():
        crop_y = int(np.where(rows)[0][0])
        crop_y2 = int(np.where(rows)[0][-1])
        crop_x = int(np.where(cols)[0][0])
        crop_x2 = int(np.where(cols)[0][-1])
    else:
        crop_y, crop_y2, crop_x, crop_x2 = 0, h, 0, w

    # Crop the image early to save memory and processing time
    gray = gray[crop_y:crop_y2, crop_x:crop_x2]
    h, w = gray.shape[:2]

    # Estimate background without using it as output
    k = max(31, min(151, (min(h, w) // 40) * 2 + 1))
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_bg)
    background = np.maximum(background, 1)

    # Normalize uneven white paper/background
    flat = cv2.divide(gray, background, scale=255)
    flat = np.clip(flat, 0, 255).astype(np.uint8)

    # Very gentle contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    contrast = clahe.apply(flat)
    base = cv2.addWeighted(flat, 0.80, contrast, 0.20, 0)

    # Soft ink map: no thresholding, no binary mask
    line_strength = (255 - base).astype(np.float32)

    # Multi-scale blackhat strengthens faint architectural strokes
    for size in (3, 5, 9, 15, 25):
        if size < min(h, w):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            blackhat = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, kernel)
            line_strength = np.maximum(line_strength, blackhat.astype(np.float32))

    # Remove only tiny scanner noise, not real edges
    line_strength = np.maximum(line_strength - 1.0, 0)

    # Gamma boost: faint gray edges become darker instead of disappearing
    ink = 255.0 * np.power(line_strength / 255.0, 0.58)
    ink *= darken_strength
    ink = np.clip(ink, 0, 255)

    result = 255.0 - ink

    # Never make any existing drawing lighter
    result = np.minimum(result, base.astype(np.float32))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Tiny sharpening, edge-preserving
    blur = cv2.GaussianBlur(result, (0, 0), 0.6)
    sharp = cv2.addWeighted(result, 1.20, blur, -0.20, 0)
    result = np.minimum(result, sharp).astype(np.uint8)

    # YOLO wants 3 channels, but all channels are identical grayscale
    yolo_input = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return result, yolo_input, crop_x, crop_y



def _add_geotiff_to_zip(zip_buffer, image_bytes,
                        tl_lat, tl_lon, tr_lat, tr_lon,
                        bl_lat, bl_lon, br_lat, br_lon,
                        bldg_x1=0, bldg_y1=0, bldg_x2=None, bldg_y2=None):
    """
    Write a geo-referenced GeoTIFF of the floor plan into the existing open zip_buffer.
    Uses rasterio if available (full GCP-based georeferencing for rotated plans),
    otherwise falls back to a simple ESRI world file (.pgw) suitable for axis-aligned plans.

    The GeoTIFF lets QGIS/ArcGIS display the floor plan perfectly overlaid on satellite
    imagery — exactly like Image 2 — without any re-vectorization.
    """
    pil_img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    
    # Make the white background transparent so it overlays cleanly on satellite imagery
    img_data = np.array(pil_img)
    white_mask = (img_data[:,:,0] > 240) & (img_data[:,:,1] > 240) & (img_data[:,:,2] > 240)
    img_data[white_mask, 3] = 0
    pil_img = PIL.Image.fromarray(img_data)
    
    W, H = pil_img.size

    # Re-open the zip for append
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        # --- write the original PNG into the zip (used by both paths) ---
        png_buf = io.BytesIO()
        pil_img.save(png_buf, format="PNG")
        zf.writestr("floorplan.png", png_buf.getvalue())

        try:
            import rasterio
            from rasterio.transform import from_gcps
            from rasterio.control import GroundControlPoint
            from rasterio.crs import CRS
            import rasterio.shutil as rio_shutil            # Build 4 GCPs: pixel (col, row)   (lon, lat)
            if bldg_x2 is None: bldg_x2 = W - 1
            if bldg_y2 is None: bldg_y2 = H - 1
            gcps = [
                GroundControlPoint(row=bldg_y1, col=bldg_x1, x=tl_lon, y=tl_lat),
                GroundControlPoint(row=bldg_y1, col=bldg_x2, x=tr_lon, y=tr_lat),
                GroundControlPoint(row=bldg_y2, col=bldg_x1, x=bl_lon, y=bl_lat),
                GroundControlPoint(row=bldg_y2, col=bldg_x2, x=br_lon, y=br_lat),
            ]
            crs = CRS.from_epsg(4326)
            
            # Calculate affine transform from GCPs so QGIS can read it natively without Warping
            transform = from_gcps(gcps)

            # Write to an in-memory GeoTIFF
            img_arr = np.array(pil_img)   # H x W x 4 (RGBA)
            tif_buf = io.BytesIO()
            with rasterio.open(
                tif_buf,
                mode="w",
                driver="GTiff",
                height=H,
                width=W,
                count=4,
                dtype=img_arr.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                for i in range(4):
                    dst.write(img_arr[:, :, i], i + 1)
                dst.update_tags(ns="rio_overview", resampling="average")

            tif_buf.seek(0)
            zf.writestr("floorplan_georef.tif", tif_buf.read())

        except ImportError:
            # rasterio not installed — write an ESRI world file instead.
            # World file assumes the plan is axis-aligned (no rotation).
            # pixel size in degrees
            px_lon = (tr_lon - tl_lon) / W
            px_lat = (bl_lat - tl_lat) / H   # negative in standard coords

            world_lines = [
                f"{px_lon:.10f}",   # A: pixel width in lon degrees
                "0.0000000000",      # D: row rotation
                "0.0000000000",      # B: col rotation
                f"{px_lat:.10f}",   # E: pixel height (negative → south)
                f"{tl_lon:.10f}",   # C: lon of top-left pixel centre
                f"{tl_lat:.10f}",   # F: lat of top-left pixel centre
            ]
            zf.writestr("floorplan.pgw", "\n".join(world_lines))
            # Also write a minimal .prj (WGS84)
            prj = ('GEOGCS["GCS_WGS_1984",'
                   'DATUM["D_WGS_1984",'
                   'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                   'PRIMEM["Greenwich",0.0],'
                   'UNIT["Degree",0.0174532925199433]]')
            zf.writestr("floorplan.prj", prj)


def main():
    """
    Main function for the Streamlit app.
    """
    # Creating sidebar
    with st.sidebar:
        st.header("Image Configuration")     # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png"))

        st.markdown("---")
        st.subheader("Image Preprocessing")
        robo_contrast = st.slider(
            "Image Contrast",
            min_value=-100,
            max_value=100,
            value=-70,
            step=10,
            help="Enhance the contrast to make faint lines more visible for detection."
        )

        robo_sharpness = st.slider(
            "Image Sharpness",
            min_value=-100,
            max_value=100,
            value=100,
            step=10,
            help="Sharpen blurry image features before detection."
        )

        # F2: always initialise these so they are defined even when no image
        # is uploaded — prevents NameError inside the button callback.
        uploaded_image = None
        processed_gray = None
        processed_img = None

        if source_img is not None:
            CLIENT = get_inference_client()
            uploaded_image = PIL.Image.open(source_img)
            # F7: pass raw bytes — @st.cache_data can hash bytes, not PIL objects
            processed_gray, processed_img, crop_x, crop_y = preprocess_floorplan(source_img.getvalue())

        st.markdown("---")
        # Multiselect for selecting labels
        available_labels = ['Door', 'Wall']
        selected_labels = setting.select_labels(available_labels)
        if 'Door' in selected_labels:
            st.markdown("---")
            st.subheader("Door Detection Settings")
            confidence = float(st.slider(
                "Door Model Confidence", 
                min_value=25, 
                max_value=100, 
                value=40,
                help="Adjust the confidence threshold for the door detection model."
            )) / 100.0
        else:
            confidence = 0.40

        if 'Wall' in selected_labels:
            st.markdown("---")
            st.subheader("Wall Detection Settings")
            wall_confidence = float(st.slider(
                "Wall Model Confidence",
                min_value=0,
                max_value=100,
                value=40,
                step=5,
                help="Adjust the confidence threshold for the wall detection model."
            )) / 100.0
        else:
            wall_confidence = 0.40

        # GIS Integration Options
        st.markdown("---")
        st.header("GIS Integration (GeoJSON)")
        enable_geojson = st.checkbox("Enable GeoJSON Export", value=False)
        if enable_geojson:
            st.info("Enter the 4 GPS coordinate corners of the floor plan for precise interpolation (handles rotation).")
            st.write("**Top Edge**")
            col_a, col_b = st.columns(2)
            tl_lat = col_a.number_input("TL Lat", value=40.712800, format="%.6f", step=0.0001)
            tl_lon = col_b.number_input("TL Lon", value=-74.006000, format="%.6f", step=0.0001)
            tr_lat = col_a.number_input("TR Lat", value=40.712800, format="%.6f", step=0.0001)
            tr_lon = col_b.number_input("TR Lon", value=-74.005500, format="%.6f", step=0.0001)

            st.write("**Bottom Edge**")
            col_c, col_d = st.columns(2)
            bl_lat = col_c.number_input("BL Lat", value=40.712500, format="%.6f", step=0.0001)
            bl_lon = col_d.number_input("BL Lon", value=-74.006000, format="%.6f", step=0.0001)
            br_lat = col_c.number_input("BR Lat", value=40.712500, format="%.6f", step=0.0001)
            br_lon = col_d.number_input("BR Lon", value=-74.005500, format="%.6f", step=0.0001)
        else:
            tl_lat = tl_lon = tr_lat = tr_lon = bl_lat = bl_lon = br_lat = br_lon = 0.0

    # Creating main page heading
    st.title("Floor Plan Object Detection using YOLOv8")

    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        if uploaded_image is not None:
              st.image(
                  uploaded_image,
                  caption="Original Image",
                  use_container_width=True
              )

        else:
            st.warning("Please upload an image.")

    model = load_model('best.pt')  # F1: uses cached loader — no reload on rerun



    if st.sidebar.button('Detect Objects'):
        if source_img is None:  # F3: reliable None-check for UploadedFile
            st.warning("Please upload an image before detecting objects.")
        else:
            # G3: warn early if Door is not selected — CSV will be empty
            if "Door" not in selected_labels:
                st.warning(
                    "\u26a0\ufe0f 'Door' is not in the selected labels. "
                    "Detection will run but the door CSV export will be empty."
                )

            spinner_text = "Detecting object (this may take a minute for massive floorplan)..." if len(selected_labels) == 1 else "Detecting objects (this may take a minute for massive floorplan)..."
            with st.spinner(spinner_text):
                
                pil_img_full = PIL.Image.open(io.BytesIO(source_img.getvalue())).convert('L')
                img_full_gray = np.array(pil_img_full, dtype=np.uint8)
                _, binary_full = cv2.threshold(img_full_gray, 240, 255, cv2.THRESH_BINARY_INV)
                kernel_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
                structural_full = cv2.morphologyEx(binary_full, cv2.MORPH_OPEN, kernel_struct)
                
                rows_struct = np.any(structural_full > 0, axis=1)
                cols_struct = np.any(structural_full > 0, axis=0)
                
                if rows_struct.any() and cols_struct.any():
                    bldg_y1 = int(np.where(rows_struct)[0][0])
                    bldg_y2 = int(np.where(rows_struct)[0][-1])
                    bldg_x1 = int(np.where(cols_struct)[0][0])
                    bldg_x2 = int(np.where(cols_struct)[0][-1])
                else:
                    W_f, H_f = pil_img_full.size
                    bldg_x1, bldg_x2, bldg_y1, bldg_y2 = 0, W_f, 0, H_f
                    
                bldg_w = bldg_x2 - bldg_x1
                bldg_h = bldg_y2 - bldg_y1

                run_walls = "Wall" in selected_labels
                run_doors = "Door" in selected_labels
                
                walls = []
                filtered_boxes = []
                
                # Base image for drawing (res_plotted)
                # Keep the coordinate offset fix: use the cropped image!
                res_plotted = processed_img.copy()
                
                if run_walls:
                    if CLIENT is not None:
                        if True: # Replaced inner spinner to use the main dynamic one
                            try:
                                # Send the original UNPROCESSED colored image to Roboflow for better accuracy,
                                # matching their online preview. We just need to crop it to match the coordinates.
                                nparr_full = np.frombuffer(source_img.getvalue(), np.uint8)
                                img_full_bgr = cv2.imdecode(nparr_full, cv2.IMREAD_COLOR)
                                h_crop, w_crop = processed_img.shape[:2]
                                raw_cropped_bgr = img_full_bgr[crop_y:crop_y + h_crop, crop_x:crop_x + w_crop]

                                # Apply User Enhancements for faint lines
                                pil_enh = PIL.Image.fromarray(cv2.cvtColor(raw_cropped_bgr, cv2.COLOR_BGR2RGB))
                                
                                # Contrast Mapping: -100 -> 0.0, 0 -> 1.0, 100 -> 2.0
                                c_factor = (robo_contrast + 100) / 100.0
                                pil_enh = ImageEnhance.Contrast(pil_enh).enhance(c_factor)
                                
                                # Sharpness Mapping: -100 -> 0.0, 0 -> 1.0, 100 -> 5.0
                                s_factor = 1.0 + (robo_sharpness / 100.0) * 4.0 if robo_sharpness > 0 else (robo_sharpness + 100) / 100.0
                                pil_enh = ImageEnhance.Sharpness(pil_enh).enhance(s_factor)

                                raw_cropped_bgr = cv2.cvtColor(np.array(pil_enh), cv2.COLOR_RGB2BGR)

                                tile_size = 1024
                                stride = 800
                                max_tiles = 250  # Increased limit for massive floorplans
                                tile_count = 0

                                for y_start in range(0, h_crop, stride):
                                    for x_start in range(0, w_crop, stride):
                                        if tile_count >= max_tiles:
                                            break
                                        
                                        y_end = min(y_start + tile_size, h_crop)
                                        x_end = min(x_start + tile_size, w_crop)
                                        
                                        tile = raw_cropped_bgr[y_start:y_end, x_start:x_end]
                                        if tile.shape[0] < 100 or tile.shape[1] < 100:
                                            continue
                                            
                                        tile_count += 1
                                        robo_res = CLIENT.infer(tile, model_id="cubicasa5k-2-qpmsa/6")
                                        
                                        for pred in robo_res.get("predictions", []):
                                            if pred.get("class", "").lower() == "wall" and pred.get("confidence", 0.0) >= wall_confidence:
                                                x_c = pred["x"] + x_start
                                                y_c = pred["y"] + y_start
                                                w_box, h_box = pred["width"], pred["height"]
                                                
                                                x1 = int(x_c - w_box / 2)
                                                y1 = int(y_c - h_box / 2)
                                                x2 = int(x_c + w_box / 2)
                                                y2 = int(y_c + h_box / 2)
                                                
                                                orient = "Horizontal" if w_box > h_box else "Vertical"
                                                walls.append({
                                                    'wall_id': len(walls) + 1,
                                                    'polygon': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                                    'length_px': max(w_box, h_box),
                                                    'thickness_px': min(w_box, h_box),
                                                    'orientation': orient,
                                                    'area_px': w_box * h_box
                                                })
                                                cv2.rectangle(res_plotted, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            except Exception as e:
                                st.error(f"Roboflow API Error: {str(e)}")
                    else:
                        st.error("Roboflow client could not be initialized. Please check your ROBOFLOW_API_KEY in .env.")
                
                if run_doors:
                    # Run local YOLO for other elements (e.g. doors, windows)
                    res = model.predict(processed_img, conf=confidence, imgsz=2048)
                    
                    # Filter boxes for non-wall objects (doors, windows, etc.)
                    filtered_boxes = [
                        box for box in res[0].boxes
                        if model.names[int(box.cls)] in selected_labels and model.names[int(box.cls)] != "Wall"
                    ]
                else:
                    filtered_boxes = []

            # G1: draw filtered detections manually instead of assigning a plain
            # Python list back to res[0].boxes (which expects a Boxes tensor).
            # This avoids a fragile internal API dependency on ultralytics internals.
            for _box in filtered_boxes:
                _lbl = model.names[int(_box.cls)]
                _bx1, _by1, _bx2, _by2 = [int(v) for v in _box.xyxy[0].tolist()]
                cv2.rectangle(res_plotted, (_bx1, _by1), (_bx2, _by2), (0, 200, 0), 5)

            # ---------------------------------
            # BUILD DOOR DATA  (F4: pandas already imported at top)
            # ---------------------------------

            door_data = []
            door_counter = 1  # F5: sequential door_id per image

            for box in filtered_boxes:
                label = model.names[int(box.cls)]

                if label == "Door":
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Add crop offset back so CSV coordinates match original CAD file
                    x1_full = x1 + crop_x
                    y1_full = y1 + crop_y
                    x2_full = x2 + crop_x
                    y2_full = y2 + crop_y

                    door_data.append({
                        "door_id": door_counter,
                        "image_name": source_img.name,
                        "label": label,
                        "x1": round(x1_full, 2),
                        "y1": round(y1_full, 2),
                        "x2": round(x2_full, 2),
                        "y2": round(y2_full, 2),
                        "center_x": round((x1_full + x2_full) / 2, 2),
                        "center_y": round((y1_full + y2_full) / 2, 2),
                        "confidence": round(float(box.conf), 4),
                    })
                    door_counter += 1

            door_df = pd.DataFrame(door_data)

            csv = door_df.to_csv(index=False).encode('utf-8')

            # ---------------------------------
            # DISPLAY RESULTS
            # ---------------------------------

            with col2:
                st.image(
                    res_plotted,
                    caption='Architecture Detected (Blue: Walls, Green: Doors)',
                    use_container_width=True
                )

                st.markdown("---")
                if run_doors:
                    st.write("Detected Doors and Locations")
                    if door_df.empty:
                        st.info("No doors were detected in this image.")
                    else:
                        st.dataframe(door_df)

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    if run_doors and not door_df.empty:
                        zip_file.writestr("door_locations.csv", csv)
                        if enable_geojson:
                            import json
                            W_full, H_full = PIL.Image.open(io.BytesIO(source_img.getvalue())).size
                            features = []
                            for _, row in door_df.iterrows():
                                def interpolate_point(px, py):
                                    u = (px - bldg_x1) / bldg_w if bldg_w else 0
                                    v = (py - bldg_y1) / bldg_h if bldg_h else 0
                                    top_lat = tl_lat + u * (tr_lat - tl_lat)
                                    top_lon = tl_lon + u * (tr_lon - tl_lon)
                                    bot_lat = bl_lat + u * (br_lat - bl_lat)
                                    bot_lon = bl_lon + u * (br_lon - bl_lon)
                                    lat = top_lat + v * (bot_lat - top_lat)
                                    lon = top_lon + v * (bot_lon - top_lon)
                                    return [lon, lat]
                                
                                pt1 = interpolate_point(row["x1"], row["y1"])
                                pt2 = interpolate_point(row["x2"], row["y1"])
                                pt3 = interpolate_point(row["x2"], row["y2"])
                                pt4 = interpolate_point(row["x1"], row["y2"])
                                poly_coords = [[pt1, pt2, pt3, pt4, pt1]]
                                
                                feature = {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": poly_coords
                                    },
                                    "properties": {
                                        "door_id": row["door_id"],
                                        "confidence": row["confidence"],
                                        "image_name": row["image_name"]
                                    }
                                }
                                features.append(feature)
                                
                            geojson_data = {
                                "type": "FeatureCollection",
                                "features": features
                            }
                            geojson_str = json.dumps(geojson_data, indent=2)
                            zip_file.writestr("door_locations.geojson", geojson_str)

                    if run_walls and walls:
                        wall_json_str = wall_detector.walls_to_json(walls)
                        zip_file.writestr("wall_locations.json", wall_json_str)
                        if enable_geojson:
                            W_full, H_full = PIL.Image.open(io.BytesIO(source_img.getvalue())).size
                            # walls are now in cropped space, so we pass (bldg_x1 - crop_x) as the offset to the GPS box
                            wall_geojson_str = wall_detector.walls_to_geojson(
                                walls=walls, crop_x=bldg_x1 - crop_x, crop_y=bldg_y1 - crop_y, crop_w=bldg_w, crop_h=bldg_h,
                                tl_lat=tl_lat, tl_lon=tl_lon, tr_lat=tr_lat, tr_lon=tr_lon,
                                bl_lat=bl_lat, bl_lon=bl_lon, br_lat=br_lat, br_lon=br_lon
                            )
                            zip_file.writestr("wall_locations.geojson", wall_geojson_str)

                if (run_doors and not door_df.empty) or (run_walls and walls):
                    # ---- add GeoTIFF to the zip if GeoJSON is enabled ----
                    if enable_geojson:
                        _add_geotiff_to_zip(zip_buffer, source_img.getvalue(),
                                            tl_lat, tl_lon, tr_lat, tr_lon,
                                            bl_lat, bl_lon, br_lat, br_lon,
                                            bldg_x1, bldg_y1, bldg_x2, bldg_y2)

                    st.download_button(
                        label="Download Detected Data (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name='detection_data.zip',
                        mime='application/zip'
                    )

                    import subprocess
                    import sys

                    # ── GN Fix #1: write predictions to disk BEFORE spawning classifier.
                    # st.download_button() only creates a browser link; it does NOT
                    # write any file.  gn_classifier.py reads door_locations.csv from
                    # disk, so we must persist it here.
                    door_df.to_csv("door_locations.csv", index=False)

                    # ── GN Fix #2 / #3 / #4: save uploaded image to disk so the
                    # classifier can auto-detect the whitespace margins and convert
                    # GT coordinates into the correct pixel space.
                    tmp_img_path = pathlib.Path("_uploaded_tmp.png")
                    tmp_img_path.write_bytes(source_img.getvalue())

                    # Run GN classifier, passing image path and dimensions
                    gn_result = subprocess.run(
                        [
                            sys.executable, "gn_classifier.py",
                            "--image-path", str(tmp_img_path),
                            "--img-w",      str(uploaded_image.width),
                            "--img-h",      str(uploaded_image.height),
                        ],
                        capture_output=True,
                        text=True
                    )

                    # ── Display GN metrics in the Streamlit UI ─────────────────
                    if run_doors and gn_result.returncode != 2:
                        st.subheader("🎯 Ground Truth Evaluation")

                        if gn_result.returncode != 0:
                            st.error(
                                "GN Classifier encountered an error:\n"
                                + (gn_result.stderr or "(no stderr output)")
                            )
                        else:
                            # Parse structured output printed by gn_classifier.py
                            gn_metrics = {}
                            for _line in gn_result.stdout.splitlines():
                                for _key in ["TP", "FP", "FN",
                                             "Precision", "Recall", "F1 Score"]:
                                    if _line.startswith(_key + ":"):
                                        gn_metrics[_key] = _line.split(":", 1)[1].strip()

                            if gn_metrics:
                                _c1, _c2, _c3 = st.columns(3)
                                _c1.metric("True Positives  (TP)",  gn_metrics.get("TP",  "—"))
                                _c2.metric("False Positives (FP)",  gn_metrics.get("FP",  "—"))
                                _c3.metric("False Negatives (FN)",  gn_metrics.get("FN",  "—"))
                                _c4, _c5, _c6 = st.columns(3)
                                _c4.metric("Precision",  gn_metrics.get("Precision",  "—"))
                                _c5.metric("Recall",     gn_metrics.get("Recall",     "—"))
                                _c6.metric("F1 Score",   gn_metrics.get("F1 Score",   "—"))

                                if pathlib.Path("door_locations_tp.csv").exists():
                                    tp_df = pd.read_csv("door_locations_tp.csv")
                                    tp_csv_bytes = tp_df.to_csv(index=False).encode("utf-8")
                                    st.download_button(
                                        label="Download TP Door Locations CSV",
                                        data=tp_csv_bytes,
                                        file_name="door_locations_tp.csv",
                                        mime="text/csv"
                                    )

                                # Run GN visualizer
                                with st.spinner("Generating Visual Evaluation Image..."):
                                    vis_result = subprocess.run(
                                        [sys.executable, "gn_visualizer.py"],
                                        capture_output=True,
                                        text=True
                                    )
                                    
                                vis_img_path = pathlib.Path("gn_visualization_result.png")
                                if vis_img_path.exists():
                                    with st.expander("View & Download Ground Truth Visualization Image"):
                                        st.image(str(vis_img_path), use_container_width=True, caption="Green: TP | Red: FP | Blue: FN")
                                        
                                        with open(vis_img_path, "rb") as f:
                                            st.download_button(
                                                label="Download GN Visualization Image",
                                                data=f.read(),
                                                file_name="gn_evaluation_result.png",
                                                mime="image/png"
                                            )
                            else:
                                st.info(
                                    "GN classifier ran but produced no parseable metrics. "
                                    "Check the console output for details."
                                )
                                st.code(gn_result.stdout)




if __name__ == "__main__":
    main()
