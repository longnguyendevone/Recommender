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

function createNumberIcon(number, variant) {
  return L.divIcon({
    className: "custom-div-icon",
    html: `<div class="poi-pin poi-pin-${variant}">${number}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -14],
  });
}

function MapViewportController({ anchor, selectedRec, recMarkerRefs }) {
  const map = useMap();

  useEffect(() => {
    if (selectedRec) {
      map.flyTo([selectedRec.lat, selectedRec.lon], 15, { duration: 0.7 });
      const marker = recMarkerRefs.current[selectedRec.poi_id];
      if (marker) {
        marker.openPopup();
      }
    }
  }, [selectedRec, map, recMarkerRefs]);

  useEffect(() => {
    if (!selectedRec && anchor) {
      map.panTo([anchor.lat, anchor.lon], { animate: true });
    }
  }, [anchor, selectedRec, map]);

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
  mapClickMode,
  onMapClick,
}) {
  const recMarkerRefs = useRef({});
  const polylinePositions = stops.map((s) => [s.lat, s.lon]);

  return (
    <div className="map-panel">
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
          recMarkerRefs={recMarkerRefs}
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
              Category: {stop.category}
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

        {showGroundTruth && groundTruth && (
          <Marker
            position={[groundTruth.lat, groundTruth.lon]}
            icon={createNumberIcon("GT", "green")}
          >
            <Popup>
              <strong>Ground Truth: {groundTruth.name}</strong>
              <br />
              Category: {groundTruth.category}
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
              <br />
              <span>{rec.explanation}</span>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
