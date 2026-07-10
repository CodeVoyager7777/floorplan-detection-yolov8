import json
def walls_to_json(walls):
    return json.dumps(walls, indent=2)


def walls_to_geojson(walls, crop_x, crop_y, crop_w, crop_h, tl_lat, tl_lon, tr_lat, tr_lon, bl_lat, bl_lon, br_lat, br_lon):
    features = []
    
    def interpolate_point(px, py):
        u = (px - crop_x) / crop_w if crop_w else 0
        v = (py - crop_y) / crop_h if crop_h else 0
        top_lat = tl_lat + u * (tr_lat - tl_lat)
        top_lon = tl_lon + u * (tr_lon - tl_lon)
        bot_lat = bl_lat + u * (br_lat - bl_lat)
        bot_lon = bl_lon + u * (br_lon - bl_lon)
        lat = top_lat + v * (bot_lat - top_lat)
        lon = top_lon + v * (bot_lon - top_lon)
        return [lon, lat]

    try:
        from shapely.geometry import Polygon, MultiPolygon, mapping
        from shapely.ops import unary_union
        
        pixel_polys = []
        for w in walls:
            pts = w['polygon']
            if len(pts) >= 3:
                pixel_polys.append(Polygon(pts))
                
        if pixel_polys:
            # Union all wall polygons
            merged = unary_union(pixel_polys)
            
            # Close tiny gaps between detections (buffer out by 15px, then buffer in by 15px)
            # join_style=2 keeps corners sharp (mitre)
            closed_walls = merged.buffer(15, join_style=2).buffer(-15, join_style=2)
            
            # Ensure it is iterable
            if isinstance(closed_walls, MultiPolygon):
                geometries = list(closed_walls.geoms)
            elif isinstance(closed_walls, Polygon):
                geometries = [closed_walls]
            else:
                geometries = []
                
            wall_id = 1
            for geom in geometries:
                # Project back to geographic coordinates
                geo_exterior = [interpolate_point(x, y) for x, y in geom.exterior.coords]
                geo_interiors = []
                for interior in geom.interiors:
                    geo_interiors.append([interpolate_point(x, y) for x, y in interior.coords])
                
                geo_poly = Polygon(geo_exterior, geo_interiors)
                
                features.append({
                    "type": "Feature",
                    "geometry": mapping(geo_poly),
                    "properties": {
                        "wall_id": wall_id,
                        "type": "Wall"
                    }
                })
                wall_id += 1
                
    except ImportError:
        # Fallback if Shapely is not available
        for w in walls:
            geo_poly = []
            for pt in w['polygon']:
                geo_pt = interpolate_point(pt[0], pt[1])
                geo_poly.append(geo_pt)
                
            if geo_poly and geo_poly[0] != geo_poly[-1]:
                geo_poly.append(geo_poly[0])
                
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_poly]
                },
                "properties": {
                    "wall_id": w['wall_id'],
                    "type": "Wall"
                }
            }
            features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return json.dumps(geojson_data, indent=2)
