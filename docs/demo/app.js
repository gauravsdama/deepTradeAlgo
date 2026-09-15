"use strict";

const list = document.querySelector("#analysis-list");
const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});
const pct = new Intl.NumberFormat("en-US", {
  style: "percent",
  signDisplay: "always",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function percent(value) {
  return pct.format(value / 100);
}

function chartPoints(series) {
  const width = 800;
  const height = 280;
  const padding = 18;
  const values = series.map((point) => point.close);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = maximum - minimum || 1;

  return series
    .map((point, index) => {
      const x = padding + (index / (series.length - 1)) * (width - padding * 2);
      const y = height - padding - ((point.close - minimum) / range) * (height - padding * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function renderGrid() {
  const grid = document.querySelector(".grid-lines");
  grid.innerHTML = [56, 112, 168, 224]
    .map((y) => `<line x1="0" y1="${y}" x2="800" y2="${y}"></line>`)
    .join("");
}

function renderEvents(events) {
  const eventList = document.querySelector("#event-list");
  eventList.replaceChildren(
    ...events.map((event) => {
      const item = document.createElement("li");
      item.innerHTML = `<time datetime="${event.date}">${event.date}</time><strong>${event.signal}</strong><span>${usd.format(event.close)}</span>`;
      return item;
    }),
  );
}

function renderAnalysis(analysis, activeButton) {
  document.querySelector("#analysis-source").textContent = analysis.source;
  document.querySelector("#ticker").textContent = analysis.ticker;
  document.querySelector("#analysis-title").textContent = analysis.title;
  document.querySelector("#analysis-prompt").textContent = analysis.prompt;
  document.querySelector("#last-close").textContent = usd.format(analysis.last_close);
  document.querySelector("#price-change").textContent = `${percent(analysis.price_change_pct)} across the saved period`;
  document.querySelector("#strategy-return").textContent = percent(analysis.strategy_return_pct);
  document.querySelector("#benchmark-return").textContent = percent(analysis.benchmark_return_pct);
  document.querySelector("#signal-count").textContent = `${analysis.buy_count} buy / ${analysis.sell_count} sell`;
  document.querySelector("#row-count").textContent = `${analysis.period.rows.toLocaleString("en-US")} analyzed rows`;
  document.querySelector("#period-start").textContent = analysis.period.start;
  document.querySelector("#period-end").textContent = analysis.period.end;
  document.querySelector("#price-line").setAttribute("points", chartPoints(analysis.series));

  const signal = document.querySelector("#signal");
  signal.className = `signal ${analysis.latest_signal.toLowerCase()}`;
  signal.querySelector("strong").textContent = analysis.latest_signal;
  document.querySelector("#chart-description").textContent =
    `${analysis.ticker} deterministic generated closing-price path from ${analysis.period.start} to ${analysis.period.end}.`;

  document.querySelectorAll(".analysis-button").forEach((button) => {
    button.classList.toggle("active", button === activeButton);
    if (button === activeButton) {
      button.setAttribute("aria-current", "true");
    } else {
      button.removeAttribute("aria-current");
    }
  });
  renderEvents(analysis.events);
}

function buildButton(analysis, index) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "analysis-button";
  button.innerHTML = `<span>${String(index + 1).padStart(2, "0")} · ${analysis.ticker}</span><strong>${analysis.title}</strong><small>${analysis.prompt}</small>`;
  button.addEventListener("click", () => renderAnalysis(analysis, button));
  return button;
}

renderGrid();
fetch("./analyses.json")
  .then((response) => {
    if (!response.ok) throw new Error(`Analysis request failed with status ${response.status}.`);
    return response.json();
  })
  .then((payload) => {
    const buttons = payload.analyses.map(buildButton);
    list.replaceChildren(...buttons);
    renderAnalysis(payload.analyses[0], buttons[0]);
  })
  .catch((error) => {
    list.innerHTML = `<p class="error">Saved analyses could not be loaded. ${error.message}</p>`;
  });
