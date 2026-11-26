import React, { useEffect, useState } from "react";
import { searchPois } from "../api/client";

const MODEL_OPTIONS = [
  { value: "hybrid", label: "Context-Aware Hybrid (Best)" },
  { value: "sequential_cl", label: "Sequential + Contrastive (CL4SRec)" },
  { value: "sequential", label: "Spatial-Temporal SASRec" },
  { value: "content_based", label: "Content-Based (TF-IDF + Spatial)" },
  { value: "cf_bpr", label: "Collaborative Filtering (BPR-MF)" },
  { value: "popularity", label: "Local & Global Popularity" },
];

export default function ControlSidebar({
  mode,
  onModeChange,
  samples,
  selectedSampleId,
  onSelectSample,
  showGroundTruth,
  onToggleGroundTruth,
  customStops,
  onAddCustomStop,
  onRemoveCustomStop,
  onClearCustomStops,
  mapClickMode,
  onToggleMapClickMode,
  customAnchor,
  model,
  onModelChange,
  maxDistanceKm,
  onMaxDistanceChange,
  hourOfDay,
  onHourChange,
  topCategories,
  selectedCategories,
  onToggleCategory,
  topK,
  onTopKChange,
  onRunRecommend,
  loading,
}) {
  const [queryText, setQueryText] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  useEffect(() => {
    if (mode !== "cold_start") {
      return undefined;
    }
    const trimmed = queryText.trim();
    if (!trimmed) {
      setSearchResults([]);
      return undefined;
    }
    const timer = setTimeout(async () => {
      try {
        const res = await searchPois(trimmed, "", 8);
        setSearchResults(res);
      } catch (err) {
        console.error(err);
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [queryText, mode]);

  return (
    <div className="sidebar-controls">
      <div className="brand-header">
        <h1>NYC Next-POI Recommender</h1>
        <p className="subtitle">HUST IT4613 | Context-Aware Prototype</p>
      </div>

      <div className="section-block">
        <label className="section-label">Demo Mode</label>
        <div className="segmented-control">
          <button
            type="button"
            className={mode === "returning" ? "seg-btn active" : "seg-btn"}
            onClick={() => onModeChange("returning")}
          >
            Returning Tourist
          </button>
          <button
            type="button"
            className={mode === "cold_start" ? "seg-btn active" : "seg-btn"}
            onClick={() => onModeChange("cold_start")}
          >
            Cold-Start Tourist
          </button>
        </div>
      </div>

      {mode === "returning" ? (
        <div className="section-block">
          <label className="section-label" htmlFor="sample-select">
            Test Trajectory ({samples.length} NYC Trails)
          </label>
          <select
            id="sample-select"
            className="input-control"
            value={selectedSampleId}
            onChange={(e) => onSelectSample(e.target.value)}
          >
            {samples.map((s) => (
              <option key={s.trail_id} value={s.trail_id}>
                User {s.user_id} | {s.stops.length} stops ({s.stops[s.stops.length - 1]?.category})
              </option>
            ))}
          </select>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={showGroundTruth}
              onChange={(e) => onToggleGroundTruth(e.target.checked)}
            />
            <span>Show Ground-Truth Next POI on Map</span>
          </label>
        </div>
      ) : (
        <div className="section-block">
          <label className="section-label" htmlFor="poi-search">
            Build Custom NYC Itinerary (1-5 stops)
          </label>
          <input
            id="poi-search"
            type="text"
            className="input-control"
            placeholder="Search NYC venues (e.g. Central Park, Museum, Coffee)..."
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
          />
          {searchResults.length > 0 && (
            <div className="autocomplete-list">
              {searchResults.map((poi) => (
                <button
                  type="button"
                  key={poi.poi_id}
                  className="autocomplete-item"
                  onClick={() => {
                    onAddCustomStop(poi);
                    setQueryText("");
                    setSearchResults([]);
                  }}
                >
                  <span className="ac-name">{poi.name}</span>
                  <span className="ac-cat">{poi.category}</span>
                </button>
              ))}
            </div>
          )}

          <div className="action-row">
            <button
              type="button"
              className={mapClickMode ? "btn-secondary active" : "btn-secondary"}
              onClick={onToggleMapClickMode}
            >
              {mapClickMode ? "Map Click Anchor: ON" : "Pick Anchor on Map"}
            </button>
            {customStops.length > 0 && (
              <button type="button" className="btn-link" onClick={onClearCustomStops}>
                Clear Stops
              </button>
            )}
          </div>

          {customAnchor && (
            <div className="anchor-pill">
              Custom Anchor: {customAnchor.lat.toFixed(4)}, {customAnchor.lon.toFixed(4)}
            </div>
          )}

          <div className="stops-list">
            {customStops.map((stop, idx) => (
              <div key={`${stop.poi_id}-${idx}`} className="stop-chip">
                <span>
                  {idx + 1}. {stop.name}
                </span>
                <button
                  type="button"
                  className="chip-remove"
                  onClick={() => onRemoveCustomStop(idx)}
                  aria-label="Remove stop"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="section-block">
        <label className="section-label" htmlFor="model-select">
          Recommender Model
        </label>
        <select
          id="model-select"
          className="input-control"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {MODEL_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      <div className="section-block grid-two">
        <div>
          <label className="section-label" htmlFor="radius-slider">
            Max Radius: {maxDistanceKm.toFixed(1)} km
          </label>
          <input
            id="radius-slider"
            type="range"
            min="0.5"
            max="15.0"
            step="0.5"
            value={maxDistanceKm}
            onChange={(e) => onMaxDistanceChange(parseFloat(e.target.value))}
          />
        </div>
        <div>
          <label className="section-label" htmlFor="topk-select">
            Top-K: {topK}
          </label>
          <select
            id="topk-select"
            className="input-control"
            value={topK}
            onChange={(e) => onTopKChange(parseInt(e.target.value, 10))}
          >
            <option value={5}>Top 5</option>
            <option value={10}>Top 10</option>
            <option value={15}>Top 15</option>
          </select>
        </div>
      </div>

      <div className="section-block">
        <label className="section-label" htmlFor="hour-select">
          Time of Day (Hour: {String(hourOfDay).padStart(2, "0")}:00)
        </label>
        <input
          id="hour-select"
          type="range"
          min="0"
          max="23"
          step="1"
          value={hourOfDay}
          onChange={(e) => onHourChange(parseInt(e.target.value, 10))}
        />
      </div>

      <div className="section-block">
        <label className="section-label">
          Category Filter ({selectedCategories.length} selected)
        </label>
        <div className="category-chips">
          {topCategories.map((cat) => {
            const active = selectedCategories.includes(cat.category);
            return (
              <button
                type="button"
                key={cat.category}
                className={active ? "cat-chip active" : "cat-chip"}
                onClick={() => onToggleCategory(cat.category)}
              >
                {cat.category}
              </button>
            );
          })}
        </div>
      </div>

      <button
        type="button"
        className="btn-primary"
        onClick={onRunRecommend}
        disabled={loading}
      >
        {loading ? "Computing Recommendations..." : "Get Top-K Recommendations"}
      </button>
    </div>
  );
}
