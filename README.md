# Baked and Happy Hot-Box

*Built for Burning Man — bake responsibly, leave no trace.*

---

## How it works

1. **Oven** — Standard electric/gas oven body; the back is rebuilt as an absorber (high absorptivity, low emissivity where it helps) sealed behind insulated double glazing so the cooking cavity stays familiar while the rear collects flux.

2. **Mirrors** — 24" x 48" Acrylic sheets are formed into **cylindrical** (single-axis curvature) reflectors. Each mirror focuses in **one direction** only; combined placement and aiming bring a stripe or patch of concentrated light onto the absorber.

3. **Tracking** — **Altitude–azimuth** mounts point each mirror at the sun. Firmware (or a control computer) computes sun position from **GPS latitude, longitude, altitude**, and **UTC time**.

4. **Temperature control** — Instead of only “on sun / off sun,” the controller **dithers** aim: briefly **on-target** (full concentration on the absorber) vs **off-target** (safe spill or sky) to hold a setpoint without overshoot, subject to available irradiance.

5. **Safety** — Concentrated light and hot surfaces are hazardous. Design for **wind**, **dust**, **mechanical failure**, and **human proximity**. Acrylic mirrors can scratch and craze; plan for inspection and spare sheets.

---

## Repository layout

This repo is the home for **mechanical drawings**, **electrical schematics**, **firmware**, **solar geometry / control notes**, and **playa logistics**. Add subfolders as the project grows, for example:

- `sim/` — optical simulations, including sun position and focused light pattern

---

**Baked and Happy Hot-Box** — sun, mirrors, and a little control theory in the desert.
