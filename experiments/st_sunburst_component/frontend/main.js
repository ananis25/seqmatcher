import { Streamlit } from "streamlit-component-lib";
import * as arrow from "apache-arrow";
import * as d3 from "d3";
import * as htl from "htl";

const app = document.querySelector("#app");
const legend = app.appendChild(document.createElement("div"));
const chart = app.appendChild(document.createElement("div"));

/** Get the hierarchy as a nested map of maps, like D3 wants it.
 *
 * @param {Array<Object>} listNodes
 */
function hierarchyFromAnchor(listNodes) {
  // we only have a subset of the full table, so we need to map full-table indices to the list indices

  let fullToListIndex = new Map();
  for (let i = 0; i < listNodes.length; i++) {
    fullToListIndex.set(listNodes[i].fullIndex, i);
  }

  function childAccessor(node) {
    let children = [];
    for (let fullIndex of node.fullChildIndices) {
      if (fullToListIndex.has(fullIndex)) {
        const listIndex = fullToListIndex.get(fullIndex);
        children.push(listNodes[listIndex]);
      }
    }
    if (children.length == 0) {
      return undefined;
    } else {
      return children;
    }
  }
  return d3.hierarchy(listNodes[0], childAccessor);
}

/**
 * Compute the X/Y coordinates for each event to render the hierarchy as a collection of arcs.
 *
 * @param {{data: {fullx0: Number, fullx1: Number, fullDepth: Number}}} nodeWrapper - event for which to draw the arc
 * @param {{data: {fullx0: Number, fullx1: Number, fullDepth: Number}}} anchorWrapper - event at the center of the chart right now
 */
function computeXY(nodeWrapper, anchorWrapper) {
  const node = nodeWrapper.data;
  const anchor = anchorWrapper.data;
  const anchorStretch = anchor.fullx1 - anchor.fullx0;
  const anchorDepth = anchor.fullDepth;
  return {
    x0:
      Math.max(0, Math.min(1, (node.fullx0 - anchor.fullx0) / anchorStretch)) *
      2 *
      Math.PI,
    x1:
      Math.max(0, Math.min(1, (node.fullx1 - anchor.fullx0) / anchorStretch)) *
      2 *
      Math.PI,
    y0: Math.max(0, node.fullDepth - anchorDepth),
    y1: Math.max(0, node.fullDepth - anchorDepth + 1),
  };
}

/**
 * Legend to show the colors mapped to each event.
 *
 * Copied off the d3 examples - https://observablehq.com/@d3/color-legend
 */
function Swatches(
  color,
  columns = null,
  swatchSize = 15,
  swatchWidth = swatchSize,
  swatchHeight = swatchSize,
  marginLeft = 0
) {
  const id = `-swatches-${Math.random().toString(16).slice(2)}`;
  const domain = color.domain();
  const format = (x) => x;

  function entity(character) {
    return `&#${character.charCodeAt(0).toString()};`;
  }

  if (columns !== null)
    return htl.html`<div style="display: flex; align-items: center; margin-left: ${+marginLeft}px; min-height: 33px; font: 10px sans-serif;">
  <style>

.${id}-item {
  break-inside: avoid;
  display: flex;
  align-items: center;
  padding-bottom: 1px;
}

.${id}-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: calc(100% - ${+swatchWidth}px - 0.5em);
}

.${id}-swatch {
  width: ${+swatchWidth}px;
  height: ${+swatchHeight}px;
  margin: 0 0.5em 0 0;
}

  </style>
  <div style=${{ width: "100%", columns }}>${domain.map((value) => {
      const label = `${format(value)}`;
      return htl.html`<div class=${id}-item>
      <div class=${id}-swatch style=${{ background: color(value) }}></div>
      <div class=${id}-label title=${label}>${label}</div>
    </div>`;
    })}
  </div>
</div>`;

  return htl.html`<div style="display: flex; align-items: center; min-height: 33px; margin-left: ${+marginLeft}px; font: 10px sans-serif;">
  <style>

.${id} {
  display: inline-flex;
  align-items: center;
  margin-right: 1em;
}

.${id}::before {
  content: "";
  width: ${+swatchWidth}px;
  height: ${+swatchHeight}px;
  margin-right: 0.5em;
  background: var(--color);
}

  </style>
  <div>${domain.map(
    (value) =>
      htl.html`<span class="${id}" style="--color: ${color(value)}">${format(
        value
      )}</span>`
  )}</div>`;
}

/**
 * A zoomable sunburst chart, to look at an aggregation of sequences of events.
 *
 * Based off the observable notebook - https://observablehq.com/@mikpanko/tennis-rallies-sunburst-chart
 */
function SunburstChart(numLevelsToPlot) {
  const availableWidth = window.innerWidth - 200;

  const diameter = (availableWidth * 3) / 5;
  const radius = diameter / 2;
  const traceBoxHeight = 30;
  const traceBoxWidth = availableWidth / 5;

  let colorScale = null;
  function setColorScale(arg) {
    colorScale = arg;
  }

  // create an svg element to display the sunburst and move it to the centre
  const svg = d3
    .select(chart)
    .append("svg")
    .attr("viewBox", [0, 0, availableWidth, diameter + 100])
    .style("font", "12px sans-serif");

  const gSunburst = svg
    .append("g")
    .attr("transform", `translate(${diameter / 2}, ${diameter / 2})`);

  // pre declare the circle group so text can overlay on top of it
  const sunburst = gSunburst.append("g").style("cursor", "pointer");
  // put the label in the centre circle
  const centerLabel = gSunburst
    .append("text")
    .attr("fill", "black")
    .attr("text-anchor", "middle")
    .style("user-select", "none")
    .attr("pointer-events", "none");
  centerLabel
    .append("tspan")
    .attr("class", "percent")
    .attr("x", 0)
    .attr("y", 0)
    .attr("font-size", "3em")
    .text(`100%`);
  centerLabel
    .append("tspan")
    .attr("class", "count")
    .attr("x", 0)
    .attr("y", 0)
    .attr("dy", "1.5em")
    .text(`(0)`);

  // arc generator
  const arc = d3
    .arc()
    .startAngle((d) => d.x0)
    .endAngle((d) => d.x1)
    .padAngle(1 / radius)
    .padRadius(radius)
    .innerRadius((d) => radius * Math.sqrt(d.y0 / numLevelsToPlot))
    .outerRadius((d) => radius * Math.sqrt(d.y1 / numLevelsToPlot) - 1);

  const gTrace = svg
    .append("g")
    .attr("transform", `translate(${diameter + 100}, 0)`);

  // initialize the mutables
  let anchorNode = null;
  let prevAnchorNode = null;
  let baseTrace = null;

  function drawChart(anchorIdx, rootToAnchorNodes, subtreeNodes, t) {
    anchorNode = hierarchyFromAnchor(subtreeNodes);

    // for the first iteration
    if (prevAnchorNode === null) {
      prevAnchorNode = anchorNode;
    }
    const nodesToPlot = anchorNode.descendants();

    sunburst
      .selectAll("path")
      .data(nodesToPlot, (d) => d.data.fullIndex)
      .join(
        (enter) =>
          enter
            .append("path")
            .attr("fill", (d) =>
              d.data.name === "root" ? "none" : colorScale(d.data.name)
            )
            .attr("fill-opacity", 0)
            .attr("pointer-events", "none")
            .on("mouseenter", onMouseEnterArc)
            .on("mouseleave", onMouseLeaveArc)
            .on("dblclick", onMouseDoubleClickArc),
        (update) => update,
        (exit) =>
          exit
            .transition(t)
            .attr("fill-opacity", 0)
            .tween("data", function (d) {
              const i = d3.interpolate(
                computeXY(d, prevAnchorNode),
                computeXY(d, anchorNode)
              );
              return function (t) {
                d3.select(this).attr("d", arc(i(t)));
              };
            })
            .remove()
      )
      .transition(t)
      .attr("fill-opacity", 1)
      .attr("pointer-events", "all")
      .tween("data", function (d) {
        const i = d3.interpolate(
          computeXY(d, prevAnchorNode),
          computeXY(d, anchorNode)
        );
        return function (t) {
          d3.select(this).attr("d", arc(i(t)));
        };
      });

    t.on("end", () => {
      centerLabel.select(".percent").text(`${anchorNode.data.percent}%`);
      centerLabel.select(".count").text(`(${anchorNode.data.count})`);

      baseTrace = [...rootToAnchorNodes].slice(1); // leave out the root node
      drawTrace(baseTrace);

      // tis the new previous
      prevAnchorNode = anchorNode;
    });
  }

  function drawTrace(trace) {
    gTrace
      .selectAll("rect")
      .data(trace, (d) => d.realIndex)
      .join("rect")
      .attr("transform", (d, i) => `translate(0, ${(i + 1) * traceBoxHeight})`)
      .attr("width", traceBoxWidth)
      .attr("height", traceBoxHeight)
      .attr("fill", (d) => colorScale(d.name))
      .attr("stroke", "white")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .attr("pointer-events", "all")
      .on("dblclick", onMouseDoubleClickBox);

    gTrace.selectAll(".trace-label").remove();
    gTrace
      .selectAll(".trace-label")
      .data(trace, (d) => d.realIndex)
      .join("text")
      .attr("class", "trace-label")
      .attr(
        "transform",
        (d, i) =>
          `translate(${traceBoxWidth / 2}, ${(i + 1.5) * traceBoxHeight})`
      )
      .attr("text-anchor", "middle")
      .attr("dy", "0.5em")
      .attr("pointer-events", "none")
      .style("user-select", "none")
      .text((d) => `${d.percent}% - ${d.name} - ${d.count}`);
  }

  function onMouseDoubleClickArc(event, ds) {
    if (ds.data.name !== "root") {
      resetAnchor(ds.data);
    }
  }

  function onMouseDoubleClickBox(event, ds) {
    resetAnchor(ds);
  }

  function resetAnchor(ds) {
    const isArcAnchor = ds.fullIndex === anchorNode.data.fullIndex;
    const newAnchorIndex = isArcAnchor ? ds.fullParentIndex : ds.fullIndex;
    Streamlit.setComponentValue(newAnchorIndex);
    // console.log("DESIRED anchor index: ", newAnchorIndex);
  }

  function onMouseEnterArc(event, ds) {
    const hoveredPath = ds.ancestors().reverse();
    sunburst
      .selectAll("path")
      .attr("fill-opacity", (d) => (hoveredPath.indexOf(d) >= 0 ? 1 : 0.3));

    centerLabel.select(".percent").text(`${ds.data.percent}%`);
    centerLabel.select(".count").text(`(${ds.data.count})`);

    // the anchor is present in both the baseTrace and hoveredPath arrays, remove it one time
    const fullTrace = [
      ...baseTrace,
      ...hoveredPath.slice(1).map((d) => d.data),
    ];
    drawTrace(fullTrace);
  }

  function onMouseLeaveArc(event) {
    sunburst.selectAll("path").attr("fill-opacity", 1);

    centerLabel.select(".percent").text(`${anchorNode.data.percent}%`);
    centerLabel.select(".count").text(`(${anchorNode.data.count})`);
    drawTrace(baseTrace);
  }

  return { svg, drawChart, setColorScale };
}

const { svg: svgChart, drawChart, setColorScale } = SunburstChart(10);

function tableToListRows(table, eventNames) {
  function unwrapList(intArray) {
    let children = [];
    for (let val of intArray) {
      children.push(val);
    }
    return children;
  }

  const out = [];
  for (let i = 0; i < table.numRows; i++) {
    const row = table.get(i);
    out.push({
      name: row.event < 0 ? "root" : eventNames[row.event],
      count: row.count,
      percent: row.percent.toPrecision(3),
      fullIndex: row.index,
      fullParentIndex: row.parent_index,
      fullDepth: row.depth,
      fullChildIndices: unwrapList(row.child_indices),
      fullx0: row.x0,
      fullx1: row.x1,
    });
  }
  return out;
}

/**
 * The component's render function. This will be called immediately after
 * the component is initially loaded, and then again every time the
 * component gets new data from Python.
 */
function onRender(event) {
  // Get the RenderData from the event
  const data = event.detail;

  // RenderData.args is the JSON dictionary of arguments sent from the Python script.
  let anchorIndex = data.args.anchorIndex;
  let eventNames = data.args.eventNames;
  let subtreeTable = arrow.tableFromIPC(data.args.subtree);
  let ancestorTable = arrow.tableFromIPC(data.args.ancestors);

  // console.log("RENDERED anchor index: ", anchorIndex);
  if (anchorIndex != subtreeTable.get(0).index) {
    throw "anchorIndex doesn't match the table sent";
  }

  const subtreeNodes = tableToListRows(subtreeTable, eventNames);
  const ancestorNodes = tableToListRows(ancestorTable, eventNames);
  const colorScale = d3
    .scaleOrdinal()
    .domain([...eventNames].sort())
    .range(
      eventNames.map((d, i) => d3.interpolateSpectral(i / eventNames.length))
    );

  legend.replaceChildren(Swatches(colorScale, "150px"));
  setColorScale(colorScale);
  drawChart(
    anchorIndex,
    ancestorNodes.reverse(),
    subtreeNodes,
    svgChart.transition().duration(1000)
  );

  // We tell Streamlit to update our frameHeight after each render event, in
  // case it has changed. (This isn't strictly necessary for the example
  // because our height stays fixed, but this is a low-cost function, so
  // there's no harm in doing it redundantly.)
  Streamlit.setFrameHeight();
}

// Attach our `onRender` handler to Streamlit's render event.
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);

// Tell Streamlit we're ready to start receiving data. We won't get our
// first RENDER_EVENT until we call this function.
Streamlit.setComponentReady();

// Finally, tell Streamlit to update our initial height. We omit the
// `height` parameter here to have it default to our scrollHeight.
Streamlit.setFrameHeight();
