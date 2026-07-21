// Flexibility engine — JS port of gridflex/models/flexibility.py.
// Site is static (no server), so this runs client-side against precomputed
// data (segments + zone_typical_demand from export_flexibility_data).

function ciOverlap(aLow, aHigh, bLow, bHigh) {
  return !(aLow > bHigh || bLow > aHigh);
}

function findBestShiftHour(segments, season, originHour, windowHours) {
  const seasonSegs = segments.filter(s => s.season === season);
  const origin = seasonSegs.find(s => s.hour === originHour);

  if (!origin) {
    return { feasible: false, reason: `origin hour ${originHour} (${season}) did not clear the quality gates — no reliable baseline rate to compare against.` };
  }

  const candidateHours = [];
  for (let h = 0; h < windowHours; h++) candidateHours.push((originHour + h) % 24);

  const candidates = seasonSegs.filter(s => candidateHours.includes(s.hour) && s.hour !== originHour);
  if (candidates.length === 0) {
    return { feasible: false, reason: `no candidate hour in the ${windowHours}h window cleared the quality gates.`, origin_rate: origin.marginal_rate_kg_per_mwh };
  }

  const defensible = candidates.filter(c =>
    !ciOverlap(c.ci95_low, c.ci95_high, origin.ci95_low, origin.ci95_high) &&
    c.marginal_rate_kg_per_mwh < origin.marginal_rate_kg_per_mwh
  );

  if (defensible.length === 0) {
    return { feasible: false, reason: `no candidate hour in the ${windowHours}h window showed a statistically defensible improvement over hour ${originHour}.`, origin_rate: origin.marginal_rate_kg_per_mwh };
  }

  const best = defensible.reduce((a, b) => a.marginal_rate_kg_per_mwh < b.marginal_rate_kg_per_mwh ? a : b);
  return {
    feasible: true,
    origin_hour: originHour,
    origin_rate: origin.marginal_rate_kg_per_mwh,
    best_hour: best.hour,
    best_rate: best.marginal_rate_kg_per_mwh,
    rate_reduction_kg_per_mwh: origin.marginal_rate_kg_per_mwh - best.marginal_rate_kg_per_mwh,
  };
}

function emissionsAvoidedKg(mw, rateOrigin, rateTarget) {
  return mw * (rateOrigin - rateTarget);
}

function valueOfShift(segments, mw, originHour, windowHours, season) {
  const shift = findBestShiftHour(segments, season, originHour, windowHours);
  if (!shift.feasible) return { feasible: false, mw, season, ...shift };
  const avoided = emissionsAvoidedKg(mw, shift.origin_rate, shift.best_rate);
  return {
    feasible: true, mw, season,
    origin_hour: shift.origin_hour, best_hour: shift.best_hour,
    origin_rate_kg_per_mwh: shift.origin_rate, best_rate_kg_per_mwh: shift.best_rate,
    emissions_avoided_kg: avoided,
  };
}

function zoneTypicalDemand(zoneDemand, zone, season, hour) {
  const row = zoneDemand.find(r => r.zone === zone && r.season === season && r.hour === hour);
  if (!row) throw new Error(`No zone_typical_demand for zone=${zone}, season=${season}, hour=${hour}`);
  return row.typical_demand_mw;
}

function zoneSeasonalPeak(zoneDemand, zone, season) {
  const rows = zoneDemand.filter(r => r.zone === zone && r.season === season);
  return Math.max(...rows.map(r => r.typical_demand_mw));
}

function peakRelief(zoneDemand, zone, mw, originHour, targetHour, season) {
  const originDemand = zoneTypicalDemand(zoneDemand, zone, season, originHour);
  const targetDemand = zoneTypicalDemand(zoneDemand, zone, season, targetHour);
  const seasonalPeak = zoneSeasonalPeak(zoneDemand, zone, season);
  return {
    zone, season, origin_hour: originHour, origin_typical_demand_mw: originDemand,
    target_hour: targetHour, target_typical_demand_mw: targetDemand,
    zone_seasonal_peak_mw: seasonalPeak,
    origin_pct_of_seasonal_peak: originDemand / seasonalPeak * 100,
    mw_shifted: mw, mw_shifted_pct_of_peak: mw / seasonalPeak * 100,
  };
}

function fullValueOfShift(segments, zoneDemand, zone, mw, originHour, windowHours, season) {
  const emissions = valueOfShift(segments, mw, originHour, windowHours, season);
  const result = { emissions };
  if (emissions.feasible) {
    result.peak_context = peakRelief(zoneDemand, zone, mw, originHour, emissions.best_hour, season);
  } else {
    result.peak_context = null;
  }
  return result;
}
