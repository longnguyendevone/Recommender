import React, { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Circle,
  MapContainer,
  Marker,
  Polyline,
  Popup,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";

const DEFAULT_CENTER = [40.7580, -73.9855];

function createNumberIcon(label, variant) {
  return L.divIcon({
    className: "custom-div-icon",
    html: `<div class="poi-pin poi-pin-${variant}">${label}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -14],
  });
}

function MapViewportController({
  anchor,
  selectedRec,
  previewPoi,
  recMarkerRefs,
  previewMarkerRef,
}) {
  const map = useMap();

  useEffect(() => {
    if (previewPoi) {
      map.flyTo([previewPoi.lat, previewPoi.lon], 15, { duration: 0.65 });
      if (previewMarkerRef.current) {
        previewMarkerRef.current.openPopup();
      }
      return;
    }
    if (selectedRec) {
      map.flyTo([selectedRec.lat, selectedRec.lon], 15, { duration: 0.65 });
      const marker = recMarkerRefs.current[selectedRec.poi_id];
      if (marker) {
        marker.openPopup();
      }
    }
  }, [selectedRec, previewPoi, map, recMarkerRefs, previewMarkerRef]);

  useEffect(() => {
    if (!selectedRec && !previewPoi && anchor) {
      map.panTo([anchor.lat, anchor.lon], { animate: true });
    }
  }, [anchor, selectedRec, previewPoi, map]);

  return null;
}

function MapClickHandler({ enabled, onMapClick }) {
  useMapEvents({
    click(e) {
      if (enabled) {
        onMapClick({ lat: e.latlng.lat, lon: e.latlng.lng });
      }
    },
  });
  return null;
}

export default function MapCanvas({
  stops,
  anchor,
  maxDistanceKm,
  recommendations,
  groundTruth,
  showGroundTruth,
  selectedRec,
  previewPoi,
  clickedPoint,
  mapClickMode,
  onMapClick,
  coldStartMode,
  onAddStopFromMap,
}) {
  const recMarkerRefs = useRef({});
  const previewMarkerRef = useRef(null);
  const polylinePositions = stops.map((s) => [s.lat, s.lon]);

  return (
    <div className={mapClickMode ? "map-panel map-pick-cursor" : "map-panel"}>
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={13}
        scrollWheelZoom
        style={{ width: "100%", height: "100vh" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapViewportController
          anchor={anchor}
          selectedRec={selectedRec}
          previewPoi={previewPoi}
          recMarkerRefs={recMarkerRefs}
          previewMarkerRef={previewMarkerRef}
        />
        <MapClickHandler enabled={mapClickMode} onMapClick={onMapClick} />

        {polylinePositions.length >= 2 && (
          <Polyline
            positions={polylinePositions}
            pathOptions={{ color: "#2563eb", weight: 3, dashArray: "6 6" }}
          />
        )}

        {stops.map((stop, idx) => (
          <Marker
            key={`stop-${stop.poi_id}-${idx}`}
            position={[stop.lat, stop.lon]}
            icon={createNumberIcon(idx + 1, "blue")}
          >
            <Popup>
              <strong>
                Stop {idx + 1}: {stop.name}
              </strong>
              <br />
              {stop.category}
              {stop.address ? (
                <>
                  <br />
                  <span className="popup-addr">{stop.address}</span>
                </>
              ) : null}
            </Popup>
          </Marker>
        ))}

        {anchor && (
          <Circle
            center={[anchor.lat, anchor.lon]}
            radius={maxDistanceKm * 1000}
            pathOptions={{
              color: "#0284c7",
              fillColor: "#38bdf8",
              fillOpacity: 0.1,
              weight: 1.5,
            }}
          />
        )}

        {clickedPoint && (
          <Marker
            position={[clickedPoint.lat, clickedPoint.lon]}
            icon={createNumberIcon("?", "amber")}
          >
            <Popup>
              <strong>Clicked Location</strong>
              <br />
              Select a nearby venue in the left panel to add it to your itinerary.
            </Popup>
          </Marker>
        )}

        {previewPoi && (
          <Marker
            position={[previewPoi.lat, previewPoi.lon]}
            icon={createNumberIcon("P", "amber")}
            ref={previewMarkerRef}
          >
            <Popup>
              <strong>{previewPoi.name}</strong>
              <br />
              {previewPoi.category}
              {previewPoi.address ? (
                <>
                  <br />
                  <span className="popup-addr">{previewPoi.address}</span>
                </>
              ) : null}
              {coldStartMode && (
                <div className="popup-action">
                  <button
                    type="button"
                    className="btn-mini btn-mini-primary"
                    onClick={() => onAddStopFromMap(previewPoi)}
                  >
                    + Add to Trail
                  </button>
                </div>
              )}
            </Popup>
          </Marker>
        )}

        {showGroundTruth && groundTruth && (
          <Marker
            position={[groundTruth.lat, groundTruth.lon]}
            icon={createNumberIcon("GT", "green")}
          >
            <Popup>
              <strong>Ground Truth: {groundTruth.name}</strong>
              <br />
              {groundTruth.category}
            </Popup>
          </Marker>
        )}

        {recommendations.map((rec) => (
          <Marker
            key={`rec-${rec.poi_id}-${rec.rank}`}
            position={[rec.lat, rec.lon]}
            icon={createNumberIcon(rec.rank, "red")}
            ref={(el) => {
              if (el) {
                recMarkerRefs.current[rec.poi_id] = el;
              }
            }}
          >
            <Popup>
              <strong>
                #{rec.rank} {rec.name}
              </strong>
              <br />
              {rec.category} | {rec.distance_km.toFixed(2)} km
              <br />
              Score: {rec.total_score.toFixed(3)}
              {coldStartMode && (
                <div className="popup-action">
                  <button
                    type="button"
                    className="btn-mini btn-mini-primary"
                    onClick={() => onAddStopFromMap(rec)}
                  >
                    + Add as Next Stop
                  </button>
                </div>
              )}
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
