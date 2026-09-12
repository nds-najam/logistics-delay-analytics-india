"""
Synthetic logistics dataset generator for DataQ Logistics Delay Intelligence PoC.

Generates a realistic Indian logistics shipment dataset with built-in causal
relationships between operational conditions (warehouse congestion, traffic,
weather, driver availability, handoffs, documentation delays, disruptions,
etc.) and delivery delays. The data is intentionally noisy so a downstream
ML model cannot simply memorize deterministic rules, and intentionally
"messy" (missing values, duplicates, outliers, inconsistent category labels)
so the pipeline can demonstrate real-world data-quality handling.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reference data: Indian cities, regions, warehouses, carriers
# --------------------------------------------------------------------------

# city -> (region, lat, lon, tier)   tier 1 = metro, 2 = large city, 3 = smaller city
CITY_INFO = {
    "Delhi":         ("North", 28.7041, 77.1025, 1),
    "Chandigarh":    ("North", 30.7333, 76.7794, 2),
    "Jaipur":        ("North", 26.9124, 75.7873, 2),
    "Lucknow":       ("North", 26.8467, 80.9462, 2),
    "Amritsar":      ("North", 31.6340, 74.8723, 3),
    "Varanasi":      ("North", 25.3176, 82.9739, 3),
    "Mumbai":        ("West", 19.0760, 72.8777, 1),
    "Ahmedabad":     ("West", 23.0225, 72.5714, 2),
    "Surat":         ("West", 21.1702, 72.8311, 2),
    "Pune":          ("West", 18.5204, 73.8567, 1),
    "Bengaluru":     ("South", 12.9716, 77.5946, 1),
    "Chennai":       ("South", 13.0827, 80.2707, 1),
    "Hyderabad":     ("South", 17.3850, 78.4867, 1),
    "Kochi":         ("South", 9.9312, 76.2673, 2),
    "Coimbatore":    ("South", 11.0168, 76.9558, 2),
    "Madurai":       ("South", 9.9252, 78.1198, 3),
    "Visakhapatnam": ("South", 17.6868, 83.2185, 2),
    "Kolkata":       ("East", 22.5726, 88.3639, 1),
    "Patna":         ("East", 25.5941, 85.1376, 2),
    "Bhubaneswar":   ("East", 20.2961, 85.8245, 2),
    "Ranchi":        ("East", 23.3441, 85.3096, 3),
    "Bhopal":        ("Central", 23.2599, 77.4126, 2),
    "Nagpur":        ("Central", 21.1458, 79.0882, 2),
    "Indore":        ("Central", 22.7196, 75.8577, 2),
    "Guwahati":      ("Northeast", 26.1445, 91.7362, 2),
}
CITIES = list(CITY_INFO.keys())
CITY_WEIGHTS = np.array([3.0 if CITY_INFO[c][3] == 1 else (1.6 if CITY_INFO[c][3] == 2 else 0.8) for c in CITIES])
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

# Warehouses: at least one per region; WH-03 (Mumbai) intentionally chronic-congested
WAREHOUSES = {
    "WH-01": {"city": "Delhi", "region": "North", "base_util": 0.72},
    "WH-02": {"city": "Chandigarh", "region": "North", "base_util": 0.58},
    "WH-03": {"city": "Mumbai", "region": "West", "base_util": 0.91},
    "WH-04": {"city": "Ahmedabad", "region": "West", "base_util": 0.64},
    "WH-05": {"city": "Bengaluru", "region": "South", "base_util": 0.79},
    "WH-06": {"city": "Chennai", "region": "South", "base_util": 0.69},
    "WH-07": {"city": "Hyderabad", "region": "South", "base_util": 0.71},
    "WH-08": {"city": "Kolkata", "region": "East", "base_util": 0.75},
    "WH-09": {"city": "Patna", "region": "East", "base_util": 0.57},
    "WH-10": {"city": "Bhopal", "region": "Central", "base_util": 0.61},
    "WH-11": {"city": "Nagpur", "region": "Central", "base_util": 0.60},
    "WH-12": {"city": "Guwahati", "region": "Northeast", "base_util": 0.55},
}
REGION_WAREHOUSES = {}
for wh, info in WAREHOUSES.items():
    REGION_WAREHOUSES.setdefault(info["region"], []).append(wh)

CARRIERS = ["Carrier A", "Carrier B", "Carrier C", "Carrier D", "Carrier E"]
CARRIER_SHARE = [0.28, 0.24, 0.20, 0.16, 0.12]
# Carrier B is intentionally the weakest performer (higher excess-hour multiplier)
CARRIER_RELIABILITY = {"Carrier A": 0.90, "Carrier B": 1.35, "Carrier C": 1.0, "Carrier D": 0.95, "Carrier E": 1.10}

SHIPPING_MODES = ["Road", "Express Road", "Rail", "Air"]
MODE_SHARE = [0.50, 0.20, 0.12, 0.18]
MODE_SPEED_KMPH = {"Road": 38, "Express Road": 52, "Rail": 50, "Air": 480}
MODE_FIXED_OVERHEAD_H = {"Road": 6, "Express Road": 4, "Rail": 8, "Air": 7}
MODE_ROUTE_FACTOR = {"Road": 1.30, "Express Road": 1.22, "Rail": 1.15, "Air": 1.05}

SERVICE_TYPES = ["Standard", "Express", "Same-Day", "Economy"]
SERVICE_SHARE = [0.50, 0.25, 0.10, 0.15]

PACKAGE_TYPES = ["Document", "Small Parcel", "Medium Parcel", "Large Parcel", "Bulk/Pallet", "Fragile"]
PACKAGE_SHARE = [0.12, 0.30, 0.28, 0.16, 0.08, 0.06]
PACKAGE_WEIGHT_PARAMS = {  # (lognormal mean, sigma) in kg
    "Document": (0.3, 0.4), "Small Parcel": (1.0, 0.5), "Medium Parcel": (2.3, 0.5),
    "Large Parcel": (3.2, 0.6), "Bulk/Pallet": (5.0, 0.7), "Fragile": (1.6, 0.6),
}

CUSTOMER_PRIORITIES = ["Standard", "Priority", "VIP"]
PRIORITY_SHARE = [0.65, 0.25, 0.10]

WEATHER_CONDITIONS = ["Clear", "Rain", "Fog", "Heavy Rain", "Storm", "Extreme Heat"]
WEATHER_SEVERITY = {"Clear": 0.0, "Rain": 1.0, "Fog": 1.5, "Heavy Rain": 2.6, "Storm": 4.0, "Extreme Heat": 1.0}

TRAFFIC_LEVELS = ["Low", "Medium", "High", "Severe"]
TRAFFIC_SEVERITY = {"Low": 0.0, "Medium": 1.0, "High": 2.0, "Severe": 3.2}

ADDRESS_QUALITY = ["Good", "Average", "Poor"]

DELAY_CAUSES = [
    "Warehouse Processing", "Pickup", "Transit", "Sorting",
    "E-Way Bill/Documentation", "Strike/Bandh Disruption", "COD Confirmation",
    "Last Mile", "Weather", "Traffic", "Address Issue", "Capacity",
    "Driver Availability", "Other",
]

DELAY_REASON_TEMPLATES = {
    "Warehouse Processing": "Origin warehouse took longer than usual to process the shipment.",
    "Pickup": "Pickup from origin was delayed due to vehicle/scheduling constraints.",
    "Transit": "Shipment experienced unusual delay while in transit.",
    "Sorting": "Extra time was lost at the sorting hub, worsened by {handoffs} handoffs.",
    "E-Way Bill/Documentation": "E-way bill / documentation was generated {eway:.1f} hours late, holding up dispatch.",
    "Strike/Bandh Disruption": "A regional strike/bandh disrupted the transit route.",
    "COD Confirmation": "COD payment/cash confirmation at delivery added extra delivery attempts.",
    "Last Mile": "Last-mile delivery took longer than expected for this shipment.",
    "Weather": "{weather} conditions (incl. monsoon impact) slowed the shipment down.",
    "Traffic": "{traffic} traffic congestion on the route slowed delivery.",
    "Address Issue": "Poor destination address quality caused delivery attempts/re-routing.",
    "Capacity": "Origin warehouse capacity utilization was at {util:.0f}%, congesting processing.",
    "Driver Availability": "Low delivery-driver availability in the destination region delayed last-mile delivery.",
    "Other": "Minor, unattributed operational variability contributed to the delay.",
}

# National holidays (approximate, repeated pattern) used to flag Holiday_Flag
HOLIDAY_MONTH_DAY = [(1, 26), (3, 8), (4, 14), (5, 1), (8, 15), (8, 19), (10, 2),
                     (10, 31), (11, 1), (11, 12), (12, 25)]

# Strike/bandh disruption windows: (region, month, day_start, day_end) — repeated yearly pattern
STRIKE_WINDOWS = [
    ("West", 3, 10, 13), ("South", 6, 5, 8), ("North", 9, 20, 23),
    ("East", 11, 15, 18), ("Central", 2, 1, 3),
]


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def _promised_days(mode, service_type, distance_km):
    tier = np.select(
        [distance_km < 200, distance_km < 500, distance_km < 1000, distance_km < 1800],
        [1, 2, 3, 4], default=5,
    ).astype(float)
    tier = tier - np.where(mode == "Air", 1, 0) - np.where(mode == "Express Road", 0.5, 0)
    tier = tier + np.select(
        [service_type == "Same-Day", service_type == "Express", service_type == "Economy"],
        [-2.0, -1.0, 1.0], default=0.0,
    )
    tier = np.clip(tier, 1, 7)
    return np.round(tier).astype(int)


def generate_synthetic_data(n_records: int = 120_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_records

    # ---- date range: ~15 months ending "today" ----
    end_date = pd.Timestamp("2026-09-01")
    start_date = end_date - pd.Timedelta(days=455)
    total_days = (end_date - start_date).days

    day_offsets = np.arange(total_days)
    dow = (start_date + pd.to_timedelta(day_offsets, unit="D")).dayofweek
    month = (start_date + pd.to_timedelta(day_offsets, unit="D")).month
    day_weight = np.where(np.isin(dow, [5, 6]), 0.65, 1.0)
    day_weight = day_weight * np.where(np.isin(month, [10, 11, 12]), 1.55, 1.0)
    day_weight = day_weight * (1.0 + 0.15 * (day_offsets / total_days))  # gentle growth trend
    day_p = day_weight / day_weight.sum()
    chosen_day = rng.choice(day_offsets, size=n, p=day_p)
    seconds_in_day = rng.integers(6 * 3600, 22 * 3600, size=n)
    order_date = start_date + pd.to_timedelta(chosen_day, unit="D") + pd.to_timedelta(seconds_in_day, unit="s")
    order_date = pd.Series(order_date)

    order_month = order_date.dt.month.to_numpy()
    order_dow = order_date.dt.dayofweek.to_numpy()
    weekend_flag = np.isin(order_dow, [5, 6]).astype(int)
    peak_season_flag = (np.isin(order_month, [10, 11, 12]) |
                         ((order_month == 1) & (order_date.dt.day.to_numpy() <= 5))).astype(int)
    holiday_flag = np.isin(
        list(zip(order_month, order_date.dt.day.to_numpy())), HOLIDAY_MONTH_DAY
    ).astype(int) if False else np.array(
        [(m, d) in HOLIDAY_MONTH_DAY for m, d in zip(order_month, order_date.dt.day.to_numpy())]
    ).astype(int)

    # ---- origin / destination ----
    origin_city = rng.choice(CITIES, size=n, p=CITY_WEIGHTS)
    destination_city = rng.choice(CITIES, size=n, p=CITY_WEIGHTS)
    same_mask = origin_city == destination_city
    while same_mask.any():
        destination_city[same_mask] = rng.choice(CITIES, size=same_mask.sum(), p=CITY_WEIGHTS)
        same_mask = origin_city == destination_city

    origin_region = np.array([CITY_INFO[c][0] for c in origin_city])
    destination_region = np.array([CITY_INFO[c][0] for c in destination_city])
    origin_tier = np.array([CITY_INFO[c][3] for c in origin_city])
    dest_tier = np.array([CITY_INFO[c][3] for c in destination_city])

    olat = np.array([CITY_INFO[c][1] for c in origin_city])
    olon = np.array([CITY_INFO[c][2] for c in origin_city])
    dlat = np.array([CITY_INFO[c][1] for c in destination_city])
    dlon = np.array([CITY_INFO[c][2] for c in destination_city])

    origin_warehouse = np.array([rng.choice(REGION_WAREHOUSES[r]) for r in origin_region])
    destination_warehouse = np.array([rng.choice(REGION_WAREHOUSES[r]) for r in destination_region])

    shipping_mode = rng.choice(SHIPPING_MODES, size=n, p=MODE_SHARE)
    carrier = rng.choice(CARRIERS, size=n, p=CARRIER_SHARE)
    service_type = rng.choice(SERVICE_TYPES, size=n, p=SERVICE_SHARE)
    package_type = rng.choice(PACKAGE_TYPES, size=n, p=PACKAGE_SHARE)
    customer_priority = rng.choice(CUSTOMER_PRIORITIES, size=n, p=PRIORITY_SHARE)

    package_weight = np.zeros(n)
    for pt in PACKAGE_TYPES:
        mask = package_type == pt
        mu, sigma = PACKAGE_WEIGHT_PARAMS[pt]
        package_weight[mask] = rng.lognormal(mean=np.log(mu), sigma=sigma, size=mask.sum())
    package_weight = np.round(np.clip(package_weight, 0.05, 60), 2)
    package_volume = np.round(package_weight * rng.uniform(0.008, 0.02, size=n) + rng.uniform(0.001, 0.01, n), 4)

    distance_km = _haversine_km(olat, olon, dlat, dlon)
    route_factor = np.array([MODE_ROUTE_FACTOR[m] for m in shipping_mode])
    distance_km = distance_km * route_factor * rng.uniform(0.95, 1.08, n)
    distance_km = np.round(np.clip(distance_km, 15, None), 1)

    # ---- COD / payment / address ----
    cod_base_p = np.select(
        [customer_priority == "Standard", customer_priority == "Priority"], [0.35, 0.15], default=0.05,
    )
    cod_flag = (rng.random(n) < cod_base_p).astype(int)
    payment_status = np.where(cod_flag == 1,
                               np.where(rng.random(n) < 0.55, "COD Pending", "COD Collected"),
                               "Paid")

    poor_addr_p = np.select([dest_tier == 1, dest_tier == 2], [0.06, 0.11], default=0.20)
    avg_addr_p = np.select([dest_tier == 1, dest_tier == 2], [0.24, 0.30], default=0.34)
    addr_roll = rng.random(n)
    address_quality = np.where(addr_roll < poor_addr_p, "Poor",
                                np.where(addr_roll < poor_addr_p + avg_addr_p, "Average", "Good"))

    # ---- weather / traffic ----
    is_monsoon_month = np.isin(order_month, [6, 7, 8, 9])
    coastal_region = np.isin(destination_region, ["West", "South", "East"])
    weather_roll = rng.random(n)
    storm_p = np.where(is_monsoon_month & coastal_region, 0.16, np.where(is_monsoon_month, 0.06, 0.02))
    heavy_rain_p = np.where(is_monsoon_month & coastal_region, 0.22, np.where(is_monsoon_month, 0.10, 0.03))
    rain_p = np.where(is_monsoon_month, 0.20, 0.08)
    fog_p = np.where(np.isin(order_month, [12, 1]) & np.isin(destination_region, ["North"]), 0.22, 0.04)
    heat_p = np.where(np.isin(order_month, [4, 5, 6]), 0.15, 0.03)
    cum = np.cumsum(np.vstack([storm_p, heavy_rain_p, rain_p, fog_p, heat_p]), axis=0)
    weather_condition = np.where(weather_roll < cum[0], "Storm",
                         np.where(weather_roll < cum[1], "Heavy Rain",
                         np.where(weather_roll < cum[2], "Rain",
                         np.where(weather_roll < cum[3], "Fog",
                         np.where(weather_roll < cum[4], "Extreme Heat", "Clear")))))

    traffic_base_p_high = np.where(origin_tier == 1, 0.30, np.where(origin_tier == 2, 0.16, 0.08))
    traffic_base_p_severe = np.where(origin_tier == 1, 0.10, 0.03) * (1 + 0.4 * peak_season_flag)
    traffic_base_p_medium = np.where(origin_tier == 1, 0.35, 0.30)
    traffic_roll = rng.random(n)
    t_cum = np.cumsum(np.vstack([traffic_base_p_severe, traffic_base_p_high, traffic_base_p_medium]), axis=0)
    traffic_level = np.where(traffic_roll < t_cum[0], "Severe",
                     np.where(traffic_roll < t_cum[1], "High",
                     np.where(traffic_roll < t_cum[2], "Medium", "Low")))

    # ---- disruption flags ----
    strike_flag = np.zeros(n, dtype=int)
    for region, mo, d0, d1 in STRIKE_WINDOWS:
        in_window = (order_month == mo) & (order_date.dt.day.to_numpy() >= d0) & (order_date.dt.day.to_numpy() <= d1)
        in_region = (origin_region == region) | (destination_region == region)
        strike_flag |= (in_window & in_region).astype(int)
    strike_flag |= (rng.random(n) < 0.004).astype(int)

    monsoon_p = np.where(is_monsoon_month & coastal_region, 0.20,
                          np.where(is_monsoon_month, 0.07, 0.01))
    monsoon_flag = (rng.random(n) < monsoon_p).astype(int)

    # ---- capacity, availability ----
    base_util = np.array([WAREHOUSES[w]["base_util"] for w in origin_warehouse])
    warehouse_capacity_utilization = base_util + 0.09 * peak_season_flag - 0.03 * weekend_flag
    warehouse_capacity_utilization += rng.normal(0, 0.045, n)
    warehouse_capacity_utilization = np.clip(warehouse_capacity_utilization, 0.30, 0.99)

    veh_low_p = 0.10 + 0.12 * peak_season_flag + 0.10 * (warehouse_capacity_utilization > 0.88)
    veh_roll = rng.random(n)
    vehicle_availability = np.where(veh_roll < veh_low_p, "Low",
                                     np.where(veh_roll < veh_low_p + 0.35, "Medium", "High"))

    drv_low_p = 0.09 + 0.10 * peak_season_flag + 0.06 * (dest_tier == 3)
    drv_roll = rng.random(n)
    driver_availability = np.where(drv_roll < drv_low_p, "Low",
                                    np.where(drv_roll < drv_low_p + 0.33, "Medium", "High"))

    number_of_handoffs = np.clip(
        rng.poisson(lam=1.6 + distance_km / 900 + np.where(shipping_mode == "Air", -0.5, 0.4)), 1, 6
    )

    # ---- e-way bill / documentation delay ----
    doc_delay_p = 0.12 + 0.10 * peak_season_flag + 0.05 * (customer_priority == "Standard") + 0.03 * cod_flag
    doc_triggered = rng.random(n) < doc_delay_p
    eway_bill_delay_hours = np.where(doc_triggered, rng.exponential(4.0, n) + 0.5, 0.0)
    eway_bill_delay_hours = np.round(eway_bill_delay_hours, 2)

    # ================================================================
    # Component hours: baseline (generic, "expected variability") +
    # named excess causes. Each named cause array also serves as the
    # candidate pool for Delay_Category attribution.
    # ================================================================
    reliability = np.array([CARRIER_RELIABILITY[c] for c in carrier])

    base_warehouse_generic = rng.gamma(shape=2.6, scale=1.3, size=n)
    peak_wh_excess = peak_season_flag * rng.uniform(0.5, 3.0, n)
    capacity_excess = np.clip(warehouse_capacity_utilization - 0.80, 0, None) * rng.uniform(40, 70, n)
    eway_excess = eway_bill_delay_hours.copy()

    base_pickup_generic = rng.gamma(shape=2.0, scale=0.8, size=n)
    vehicle_excess = np.where(vehicle_availability == "Low", rng.uniform(1.0, 5.0, n),
                       np.where(vehicle_availability == "Medium", rng.uniform(0.0, 1.5, n), 0.0))

    base_transit_expected = MODE_FIXED_OVERHEAD_H[shipping_mode[0]] if False else np.array(
        [MODE_FIXED_OVERHEAD_H[m] for m in shipping_mode]
    ) + distance_km / np.array([MODE_SPEED_KMPH[m] for m in shipping_mode])
    transit_generic = rng.gamma(shape=2.0, scale=1.1, size=n)
    weather_sev = np.array([WEATHER_SEVERITY[w] for w in weather_condition])
    weather_excess = weather_sev * rng.uniform(1.0, 3.0, n)
    monsoon_excess = monsoon_flag * rng.uniform(3.0, 15.0, n)
    weather_cause_hours = weather_excess + 0.6 * monsoon_excess
    strike_excess = strike_flag * rng.uniform(6.0, 30.0, n)
    traffic_sev = np.array([TRAFFIC_SEVERITY[t] for t in traffic_level])
    traffic_excess_total = traffic_sev * rng.uniform(0.7, 2.3, n)
    traffic_excess_transit = traffic_excess_total * 0.35
    traffic_excess_lastmile = traffic_excess_total * 0.65 + 0.4 * monsoon_excess

    base_sorting_generic = rng.gamma(shape=1.8, scale=1.1, size=n)
    handoff_excess = np.clip(number_of_handoffs - 2, 0, None) * rng.uniform(0.8, 2.2, n)

    base_lastmile_generic = rng.gamma(shape=2.3, scale=1.6, size=n)
    driver_excess = np.where(driver_availability == "Low", rng.uniform(2.0, 9.0, n),
                      np.where(driver_availability == "Medium", rng.uniform(0.0, 2.0, n), 0.0))
    address_excess = np.where(address_quality == "Poor", rng.uniform(2.0, 11.0, n),
                       np.where(address_quality == "Average", rng.uniform(0.0, 2.5, n), 0.0))
    cod_excess = cod_flag * rng.uniform(1.0, 6.5, n)

    other_noise = rng.uniform(0.0, 3.2, n)

    # assemble stage totals (apply carrier reliability multiplier to excess only)
    excess_multiplier = reliability
    warehouse_processing_hours = (base_warehouse_generic + peak_wh_excess) + \
        (capacity_excess + eway_excess) * excess_multiplier + 0.5 * other_noise
    pickup_delay_hours = base_pickup_generic + vehicle_excess * excess_multiplier
    transit_hours = base_transit_expected + transit_generic + \
        (weather_cause_hours + strike_excess + traffic_excess_transit) * excess_multiplier
    sorting_hours = base_sorting_generic + handoff_excess * excess_multiplier
    last_mile_hours = base_lastmile_generic + \
        (driver_excess + address_excess + cod_excess + traffic_excess_lastmile) * excess_multiplier + 0.5 * other_noise

    warehouse_processing_hours = np.clip(warehouse_processing_hours, 0.2, None)
    pickup_delay_hours = np.clip(pickup_delay_hours, 0.1, None)
    transit_hours = np.clip(transit_hours, 1.0, None)
    sorting_hours = np.clip(sorting_hours, 0.1, None)
    last_mile_hours = np.clip(last_mile_hours, 0.2, None)

    total_actual_hours = warehouse_processing_hours + pickup_delay_hours + transit_hours + sorting_hours + last_mile_hours

    pickup_date = order_date + pd.to_timedelta(pickup_delay_hours, unit="h")
    actual_delivery_date = pickup_date + pd.to_timedelta(
        warehouse_processing_hours + transit_hours + sorting_hours + last_mile_hours, unit="h"
    )

    promised_days = _promised_days(shipping_mode, service_type, distance_km)
    expected_delivery_date = (order_date.dt.normalize() + pd.to_timedelta(promised_days, unit="D") +
                              pd.Timedelta(hours=10))

    delay_hours_raw = (actual_delivery_date - expected_delivery_date).dt.total_seconds() / 3600.0
    delay_flag = (delay_hours_raw > 0).astype(int)
    delay_hours = np.clip(delay_hours_raw, 0, None)
    actual_delivery_days = (actual_delivery_date - order_date).dt.total_seconds() / 86400.0

    # ---- delay category attribution (argmax over named cause buckets) ----
    cause_matrix = np.vstack([
        base_warehouse_generic + peak_wh_excess,          # Warehouse Processing
        base_pickup_generic + vehicle_excess,             # Pickup
        transit_generic,                                  # Transit
        base_sorting_generic,                             # Sorting
        eway_excess,                                       # E-Way Bill/Documentation
        strike_excess,                                     # Strike/Bandh Disruption
        cod_excess,                                        # COD Confirmation
        base_lastmile_generic,                             # Last Mile
        weather_cause_hours,                               # Weather
        traffic_excess_transit + traffic_excess_lastmile,  # Traffic
        address_excess,                                    # Address Issue
        capacity_excess,                                   # Capacity
        driver_excess,                                     # Driver Availability
        other_noise,                                       # Other
    ])
    cause_idx = np.argmax(cause_matrix, axis=0)
    delay_category = np.where(delay_flag == 1, np.array(DELAY_CAUSES)[cause_idx], "On Time")

    delay_reason = np.empty(n, dtype=object)
    for i in range(n):
        if delay_flag[i] == 0:
            delay_reason[i] = "Delivered within promised window."
        else:
            cat = delay_category[i]
            tmpl = DELAY_REASON_TEMPLATES[cat]
            delay_reason[i] = tmpl.format(
                handoffs=number_of_handoffs[i], eway=eway_bill_delay_hours[i],
                weather=weather_condition[i], traffic=traffic_level[i],
                util=warehouse_capacity_utilization[i] * 100,
            )

    shipment_id = np.array([f"SH{100000 + i}" for i in range(n)])
    order_id = np.array([f"OD{100000 + i}" for i in range(n)])
    n_customers = max(1000, n // 3)
    customer_pool = np.array([f"CUST{10000 + i}" for i in range(n_customers)])
    customer_weights = rng.zipf(1.4, n_customers).astype(float)
    customer_weights = customer_weights / customer_weights.sum()
    customer_id = rng.choice(customer_pool, size=n, p=customer_weights)

    df = pd.DataFrame({
        "Shipment_ID": shipment_id, "Order_ID": order_id, "Customer_ID": customer_id,
        "Origin_City": origin_city, "Origin_Region": origin_region,
        "Destination_City": destination_city, "Destination_Region": destination_region,
        "Origin_Warehouse": origin_warehouse, "Destination_Warehouse": destination_warehouse,
        "Shipping_Mode": shipping_mode, "Carrier": carrier, "Service_Type": service_type,
        "Package_Type": package_type, "Package_Weight": package_weight, "Package_Volume": package_volume,
        "Order_Date": order_date, "Pickup_Date": pickup_date,
        "Expected_Delivery_Date": expected_delivery_date, "Actual_Delivery_Date": actual_delivery_date,
        "Promised_Delivery_Days": promised_days, "Actual_Delivery_Days": np.round(actual_delivery_days, 2),
        "Distance_KM": distance_km,
        "Warehouse_Processing_Time_Hours": np.round(warehouse_processing_hours, 2),
        "Pickup_Delay_Hours": np.round(pickup_delay_hours, 2),
        "Transit_Time_Hours": np.round(transit_hours, 2),
        "Sorting_Time_Hours": np.round(sorting_hours, 2),
        "Last_Mile_Time_Hours": np.round(last_mile_hours, 2),
        "Number_of_Handoffs": number_of_handoffs,
        "Weather_Condition": weather_condition, "Traffic_Level": traffic_level,
        "Holiday_Flag": holiday_flag, "Weekend_Flag": weekend_flag, "Peak_Season_Flag": peak_season_flag,
        "Vehicle_Availability": vehicle_availability, "Driver_Availability": driver_availability,
        "Warehouse_Capacity_Utilization": np.round(warehouse_capacity_utilization * 100, 1),
        "EWay_Bill_Delay_Hours": eway_bill_delay_hours,
        "Strike_Disruption_Flag": strike_flag, "COD_Flag": cod_flag, "Monsoon_Disruption_Flag": monsoon_flag,
        "Payment_Status": payment_status, "Address_Quality": address_quality,
        "Customer_Priority": customer_priority,
        "Delay_Flag": delay_flag, "Delay_Hours": np.round(delay_hours, 2),
        "Delay_Category": delay_category, "Delay_Reason": delay_reason,
    })
    return df


def inject_data_quality_issues(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Introduce a controlled amount of missing values, duplicates, outliers,
    and inconsistent categorical labels to simulate real-world data mess."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # --- missing values (1-4% in a handful of columns) ---
    missing_cols = {
        "Package_Weight": 0.02, "Address_Quality": 0.015, "Driver_Availability": 0.02,
        "Weather_Condition": 0.015, "Traffic_Level": 0.015, "Vehicle_Availability": 0.02,
        "Customer_Priority": 0.01, "Actual_Delivery_Date": 0.005,
    }
    for col, frac in missing_cols.items():
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # --- inconsistent categorical labels ---
    def messy_variants(series, variant_map, frac):
        idx = rng.choice(n, size=int(n * frac), replace=False)
        s = series.copy()
        for i in idx:
            val = s.iat[i]
            if val in variant_map:
                s.iat[i] = rng.choice(variant_map[val])
        return s

    city_variants = {
        "Mumbai": ["mumbai", "MUMBAI", " Mumbai", "Bombay"],
        "Bengaluru": ["Bangalore", "bengaluru", "BENGALURU"],
        "Delhi": ["New Delhi", "delhi", "DELHI "],
        "Kochi": ["Cochin", "kochi"],
    }
    df["Origin_City"] = messy_variants(df["Origin_City"], city_variants, 0.02)
    df["Destination_City"] = messy_variants(df["Destination_City"], city_variants, 0.02)

    carrier_variants = {
        "Carrier A": ["carrier a", "CARRIER A", "Carrier-A"],
        "Carrier B": ["carrier b", "Carrier_B"],
        "Carrier C": ["carrier c"],
    }
    df["Carrier"] = messy_variants(df["Carrier"], carrier_variants, 0.02)

    # --- outliers ---
    out_idx = rng.choice(n, size=int(n * 0.003), replace=False)
    df.loc[out_idx, "Package_Weight"] = rng.uniform(500, 5000, size=len(out_idx))
    out_idx2 = rng.choice(n, size=int(n * 0.002), replace=False)
    df.loc[out_idx2, "Distance_KM"] = rng.uniform(20000, 90000, size=len(out_idx2))
    out_idx3 = rng.choice(n, size=int(n * 0.002), replace=False)
    df.loc[out_idx3, "Delay_Hours"] = df.loc[out_idx3, "Delay_Hours"] + rng.uniform(500, 2000, size=len(out_idx3))

    # --- duplicate records ---
    dup_frac = 0.007
    dup_idx = rng.choice(n, size=int(n * dup_frac), replace=False)
    dup_rows = df.iloc[dup_idx]
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def build_and_save(n_records: int = 120_000, seed: int = 42, out_path: str = None) -> pd.DataFrame:
    df = generate_synthetic_data(n_records=n_records, seed=seed)
    df = inject_data_quality_issues(df, seed=seed + 1)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    if out_path:
        df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logistics_data.csv")
    out = os.path.abspath(out)
    data = build_and_save(n_records=120_000, seed=42, out_path=out)
    print(f"Generated {len(data):,} records -> {out}")
    print("\nDelay rate:", round(data["Delay_Flag"].mean() * 100, 2), "%")
    print("\nDelay category distribution (delayed only):")
    print(data.loc[data["Delay_Flag"] == 1, "Delay_Category"].value_counts(normalize=True).round(3) * 100)
    print("\nColumns:", len(data.columns))
