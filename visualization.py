# src/visualization.py
import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {
            'team_a': (0, 0, 255),    # Red (BGR format)
            'team_b': (255, 0, 0),    # Blue
            'ball': (0, 255, 0),      # Green
            'umpire': (0, 255, 255),  # Yellow
            'unknown': (128, 128, 128) # Gray
        }
    
    def draw_tracking(self, frame, tracks, team_info):
        """Draw tracking information on frame"""
        annotated_frame = frame.copy()
        
        if not tracks:
            return annotated_frame
        
        for track in tracks:
            try:
                # Check if track is confirmed
                if not hasattr(track, 'is_confirmed'):
                    continue
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                
                # Get bounding box
                if hasattr(track, 'to_ltrb'):
                    bbox = track.to_ltrb()
                elif hasattr(track, 'to_tlbr'):
                    bbox = track.to_tlbr()
                else:
                    continue
                
                # Ensure bbox is a list/tuple of 4 elements
                if not isinstance(bbox, (list, tuple, np.ndarray)):
                    continue
                if len(bbox) != 4:
                    continue
                
                # Convert to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get class name
                class_name = 'player'  # Default
                if hasattr(track, 'get_class'):
                    class_name = track.get_class() or 'player'
                
                # Get team info
                team = team_info.get(str(track_id), 'unknown')
                color = self.colors.get(team, self.colors['unknown'])
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}_{track_id}"
                if team != 'unknown':
                    label = f"{team}_{label}"
                
                cv2.putText(annotated_frame, label, (x1, max(y1-10, 0)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            except Exception as e:
                # Skip this track if there's an error
                continue
        
        return annotated_frame
    
    def create_tactical_map(self, field_template, positions, frame_idx):
        """Create Hawk-Eye style tactical map"""
        if field_template is None:
            # Create a simple template
            field_template = np.zeros((600, 800, 3), dtype=np.uint8)
            field_template[:] = (34, 139, 34)  # Green
        
        tactical_map = field_template.copy()
        
        if not positions:
            return tactical_map
        
        frame_key = str(frame_idx)
        
        for entity_id, data in positions.items():
            try:
                if frame_key not in data.get('positions', {}):
                    continue
                
                pos = data['positions'][frame_key]
                if not isinstance(pos, (list, tuple, np.ndarray)) or len(pos) < 2:
                    continue
                
                team = data.get('team', 'unknown')
                color = self.colors.get(team, self.colors['unknown'])
                
                # Scale position to image coordinates
                x = int(pos[0] * tactical_map.shape[1] / 1280)  # Scale to width
                y = int(pos[1] * tactical_map.shape[0] / 720)   # Scale to height
                
                # Ensure coordinates are within bounds
                x = max(0, min(x, tactical_map.shape[1] - 1))
                y = max(0, min(y, tactical_map.shape[0] - 1))
                
                # Draw player/entity on map
                if data.get('class') == 'ball':
                    cv2.circle(tactical_map, (x, y), 8, color, -1)
                else:
                    cv2.circle(tactical_map, (x, y), 6, color, -1)
                    # Draw track ID
                    cv2.putText(tactical_map, str(entity_id), (x-5, y+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            except Exception as e:
                continue
        
        return tactical_map