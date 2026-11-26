import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchCategories,
  fetchSampleUsers,
  postRecommend,
} from "./api/client";
import ControlSidebar from "./components/ControlSidebar";
import MapCanvas from "./components/MapCanvas";
import RecommendationList from "./components/RecommendationList";

const DEFAULT_ANCHOR = { lat: 40.7580, lon: -73.9855 };

export default function App() {
  const [mode, setMode] = useState("returning");
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [showGroundTruth, setShowGroundTruth] = useState(true);

  const [customStops, setCustomStops] = useState([]);
  const [mapClickMode, setMapClickMode] = useState(false);
  const [customAnchor, setCustomAnchor] = useState(null);

  const [topCategories, setTopCategories] = useState([]);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [model, setModel] = useState("hybrid");
  const [maxDistanceKm, setMaxDistanceKm] = useState(5.0);
  const [hourOfDay, setHourOfDay] = useState(14);
  const [topK, setTopK] = useState(10);

  const [recommendations, setRecommendations] = useState([]);
  const [responseMeta, setResponseMeta] = useState(null);
  const [selectedRec, setSelectedRec] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    async function init() {
      try {
        const [sampleList, catList] = await Promise.all([
          fetchSampleUsers(20),
          fetchCategories(),
        ]);
        setSamples(sampleList);
        if (sampleList.length > 0) {
          setSelectedSampleId(sampleList[0].trail_id);
        }
        setTopCategories(catList.slice(0, 18));
      } catch (err) {
        setErrorMsg(`Failed to connect to FastAPI backend: ${err.message}`);
      }
    }
    init();
  }, []);

  const currentSample = useMemo(
    () => samples.find((s) => s.trail_id === selectedSampleId) || null,
    [samples, selectedSampleId]
  );

  const activeStops = useMemo(() => {
    if (mode === "returning") {
      return currentSample ? currentSample.stops : [];
    }
    return customStops;
  }, [mode, currentSample, customStops]);

  const activeAnchor = useMemo(() => {
    if (mode === "cold_start" && customAnchor) {
      return customAnchor;
    }
    if (activeStops.length > 0) {
      const last = activeStops[activeStops.length - 1];
      return { lat: last.lat, lon: last.lon };
    }
    return DEFAULT_ANCHOR;
  }, [mode, customAnchor, activeStops]);

  const runRecommend = useCallback(async () => {
    setLoading(true);
    setErrorMsg("");
    setSelectedRec(null);
    try {
      const payload = {
        user_id: mode === "returning" && currentSample ? currentSample.user_id : null,
        trajectory: activeStops.map((s) => s.poi_id),
        current_lat: activeAnchor.lat,
        current_lon: activeAnchor.lon,
        hour_of_day: hourOfDay,
        max_distance_km: maxDistanceKm,
        categories: selectedCategories,
        model,
        top_k: topK,
      };
      const res = await postRecommend(payload);
      setRecommendations(res.recommendations);
      setResponseMeta({
        model_used: res.model_used,
        inference_time_ms: res.inference_time_ms,
        is_cold_start: res.is_cold_start,
      });
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  }, [
    mode,
    currentSample,
    activeStops,
    activeAnchor,
    hourOfDay,
    maxDistanceKm,
    selectedCategories,
    model,
    topK,
  ]);

  useEffect(() => {
    if (mode === "returning" && currentSample) {
      runRecommend();
    }
  }, [mode, currentSample, model, maxDistanceKm, topK, selectedCategories, hourOfDay, runRecommend]);

  const handleToggleCategory = (cat) => {
    setSelectedCategories((prev) =>
      prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
    );
  };

  const handleAddCustomStop = (poi) => {
    setCustomStops((prev) => [...prev.slice(-4), poi]);
    setCustomAnchor(null);
  };

  const handleRemoveCustomStop = (idx) => {
    setCustomStops((prev) => prev.filter((_, i) => i !== idx));
  };

  return (
    <div className="app-shell">
      <aside className="left-pane">
        <ControlSidebar
          mode={mode}
          onModeChange={(nextMode) => {
            setMode(nextMode);
            setSelectedRec(null);
          }}
          samples={samples}
          selectedSampleId={selectedSampleId}
          onSelectSample={(id) => {
            setSelectedSampleId(id);
            setSelectedRec(null);
          }}
          showGroundTruth={showGroundTruth}
          onToggleGroundTruth={setShowGroundTruth}
          customStops={customStops}
          onAddCustomStop={handleAddCustomStop}
          onRemoveCustomStop={handleRemoveCustomStop}
          onClearCustomStops={() => {
            setCustomStops([]);
            setCustomAnchor(null);
          }}
          mapClickMode={mapClickMode}
          onToggleMapClickMode={() => setMapClickMode((v) => !v)}
          customAnchor={customAnchor}
          model={model}
          onModelChange={setModel}
          maxDistanceKm={maxDistanceKm}
          onMaxDistanceChange={setMaxDistanceKm}
          hourOfDay={hourOfDay}
          onHourChange={setHourOfDay}
          topCategories={topCategories}
          selectedCategories={selectedCategories}
          onToggleCategory={handleToggleCategory}
          topK={topK}
          onTopKChange={setTopK}
          onRunRecommend={runRecommend}
          loading={loading}
        />

        {errorMsg && <div className="error-banner">{errorMsg}</div>}

        <RecommendationList
          responseMeta={responseMeta}
          recommendations={recommendations}
          selectedRec={selectedRec}
          onSelectRec={setSelectedRec}
        />
      </aside>

      <main className="right-pane">
        <MapCanvas
          stops={activeStops}
          anchor={activeAnchor}
          maxDistanceKm={maxDistanceKm}
          recommendations={recommendations}
          groundTruth={mode === "returning" && currentSample ? currentSample.ground_truth : null}
          showGroundTruth={mode === "returning" && showGroundTruth}
          selectedRec={selectedRec}
          mapClickMode={mode === "cold_start" && mapClickMode}
          onMapClick={(coords) => {
            setCustomAnchor(coords);
            setMapClickMode(false);
          }}
        />
      </main>
    </div>
  );
}
