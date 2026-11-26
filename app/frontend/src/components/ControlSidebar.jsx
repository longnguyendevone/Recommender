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
  onLocatePoi,
  mapClickMode,
  onToggleMapClickMode,
  nearbyPois,
  nearbyLoading,
  clickedPoint,
  onUseClickedAsAnchor,
  onDismissNearby,
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
  const [filtersOpen, setFiltersOpen] = useState(false);

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

  const filterSummary = `${maxDistanceKm.toFixed(1)} km | ${String(hourOfDay).padStart(
    2,
    "0"
  )}:00 | Top ${topK} | ${
    selectedCategories.length === 0
      ? "All categories"
      : `${selectedCategories.length} cat`
  }`;

  return (
    <div className="sidebar-controls">
      <div className="brand-header">
        <div>
          <h1>NYC Next-POI Recommender</h1>
          <p className="subtitle">HUST IT4613 | Context-Aware Prototype</p>
        </div>
      </div>

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

      {mode === "returning" ? (
        <div className="panel-card">
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
                User {s.user_id} | {s.stops.length} stops (
                {s.stops[s.stops.length - 1]?.category})
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
        <div className="panel-card">
          <div className="row-between">
            <label className="section-label" htmlFor="poi-search">
              Initial POIs ({customStops.length}/5)
            </label>
            <div className="inline-actions">
              <button
                type="button"
                className={mapClickMode ? "btn-pill active" : "btn-pill"}
                onClick={onToggleMapClickMode}
              >
                {mapClickMode ? "Map Pick: ON" : "Map Pick: OFF"}
              </button>
              {customStops.length > 0 && (
                <button
                  type="button"
                  className="btn-link"
                  onClick={onClearCustomStops}
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          <div className="search-wrap">
            <input
              id="poi-search"
              type="text"
              className="input-control"
              placeholder="Search NYC venues (e.g. Starbucks, Central Park, Museum)..."
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
            />
            {queryText && (
              <button
                type="button"
                className="search-clear"
                onClick={() => {
                  setQueryText("");
                  setSearchResults([]);
                }}
                aria-label="Clear search"
              >
                x
              </button>
            )}
          </div>

          {searchResults.length > 0 && (
            <div className="poi-picker-list">
              {searchResults.map((poi) => (
                <div key={poi.poi_id} className="poi-picker-row">
                  <div className="poi-picker-info">
                    <div className="poi-picker-title">
                      <span className="ac-name">{poi.name}</span>
                      <span className="ac-cat">{poi.category}</span>
                    </div>
                    <div className="poi-picker-addr">
                      {poi.address || "New York, NY"}
                    </div>
                  </div>
                  <div className="poi-picker-actions">
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={() => onLocatePoi(poi)}
                    >
                      Locate
                    </button>
                    <button
                      type="button"
                      className="btn-mini btn-mini-primary"
                      onClick={() => {
                        onAddCustomStop(poi);
                        setQueryText("");
                        setSearchResults([]);
                      }}
                    >
                      + Add
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {(nearbyLoading || (clickedPoint && nearbyPois.length > 0)) && (
            <div className="nearby-box">
              <div className="row-between">
                <span className="nearby-title">
                  {nearbyLoading
                    ? "Finding nearby NYC venues..."
                    : "Nearby Clicked Map Point"}
                </span>
                <div className="inline-actions">
                  {clickedPoint && (
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={onUseClickedAsAnchor}
                    >
                      Anchor Only
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn-link-muted"
                    onClick={onDismissNearby}
                  >
                    Hide
                  </button>
                </div>
              </div>
              {!nearbyLoading && (
                <div className="poi-picker-list compact">
                  {nearbyPois.map((poi) => (
                    <div key={`nb-${poi.poi_id}`} className="poi-picker-row">
                      <div className="poi-picker-info">
                        <div className="poi-picker-title">
                          <span className="ac-name">{poi.name}</span>
                          <span className="ac-cat">
                            {poi.category} | {poi.distance_km.toFixed(2)} km
                          </span>
                        </div>
                        <div className="poi-picker-addr">
                          {poi.address || "New York, NY"}
                        </div>
                      </div>
                      <div className="poi-picker-actions">
                        <button
                          type="button"
                          className="btn-mini"
                          onClick={() => onLocatePoi(poi)}
                        >
                          Locate
                        </button>
                        <button
                          type="button"
                          className="btn-mini btn-mini-primary"
                          onClick={() => onAddCustomStop(poi)}
                        >
                          + Add
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {customStops.length > 0 ? (
            <div className="stops-list">
              {customStops.map((stop, idx) => (
                <div key={`${stop.poi_id}-${idx}`} className="stop-chip">
                  <button
                    type="button"
                    className="stop-chip-label"
                    onClick={() => onLocatePoi(stop)}
                    title={stop.address}
                  >
                    {idx + 1}. {stop.name}
                  </button>
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
          ) : (
            <div className="hint-text">
              Search a venue or click anywhere on the NYC map to pick initial stops.
            </div>
          )}

          {customAnchor && (
            <div className="anchor-pill">
              Custom Anchor: {customAnchor.lat.toFixed(4)},{" "}
              {customAnchor.lon.toFixed(4)}
            </div>
          )}
        </div>
      )}

      <div className="compact-model-row">
        <div className="model-select-wrap">
          <label className="section-label" htmlFor="model-select">
            Model
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
        <button
          type="button"
          className="btn-primary btn-run-compact"
          onClick={onRunRecommend}
          disabled={loading}
        >
          {loading ? "Running..." : "Recommend"}
        </button>
      </div>

      <div className="collapsible-box">
        <button
          type="button"
          className="collapsible-trigger"
          onClick={() => setFiltersOpen((v) => !v)}
        >
          <span className="section-label">Filters & Constraints</span>
          <span className="collapse-summary">{filterSummary}</span>
          <span className="collapse-caret">{filtersOpen ? "-" : "+"}</span>
        </button>

        {filtersOpen && (
          <div className="collapsible-body">
            <div className="grid-two">
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
                  onChange={(e) =>
                    onMaxDistanceChange(parseFloat(e.target.value))
                  }
                />
              </div>
              <div>
                <label className="section-label" htmlFor="topk-select">
                  Top-K
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

            <div>
              <label className="section-label" htmlFor="hour-select">
                Time of Day ({String(hourOfDay).padStart(2, "0")}:00)
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

            <div>
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
          </div>
        )}
      </div>
    </div>
  );
}
