<script>
    import { onMount, tick } from "svelte";
    import * as d3 from "d3";

    export let data = {};
    export let title = "";

    let container;
    let legendContainer;
    let tooltip;
    let selectedIndex = null;

    async function drawPieChart() {
        if (!data || Object.keys(data).length === 0) return;

        await tick();

        d3.select(container).selectAll("*").remove();
        d3.select(legendContainer).selectAll("*").remove();

        const width = 300;
        const height = 300;
        const padding = 20;
        const radius = Math.min(width, height) / 2 - padding;

        const color = d3.scaleOrdinal(d3.schemeTableau10);
        const pie = d3.pie().value(d => d[1]);
        const data_ready = pie(Object.entries(data));

        const arc = d3.arc().innerRadius(0).outerRadius(radius);
        const arcHover = d3.arc().innerRadius(0).outerRadius(radius * 1.10);

        const svg = d3.select(container)
            .append("svg")
            .attr("width", width + padding * 2)
            .attr("height", height + padding * 2)
            .attr("viewBox", `-${width / 2 + padding} -${height / 2 + padding} ${width + padding * 2} ${height + padding * 2}`)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .style("overflow", "visible")
            .append("g")
            .attr("transform", `translate(0, 0)`);

        tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);

        const slices = svg.selectAll("path")
            .data(data_ready)
            .enter()
            .append("path")
            .attr("d", arc)
            .attr("fill", d => color(d.data[0]))
            .attr("stroke", "#fff")
            .style("stroke-width", "2px")
            .style("cursor", "pointer")
            .on("mouseover", function (event, d) {
                d3.select(this).transition().duration(200).attr("d", arcHover);
                tooltip.transition().duration(200).style("opacity", 1);
                tooltip.html(`<strong>${d.data[0]}</strong>: ${d.data[1]} (${((d.data[1] / d3.sum(Object.values(data))) * 100).toFixed(1)}%)`)
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY - 20}px`);
            })
            .on("mouseout", function () {
                d3.select(this).transition().duration(200).attr("d", arc);
                tooltip.transition().duration(200).style("opacity", 0);
            })
            .on("click", function (_, d) {
                selectedIndex = selectedIndex === d.index ? null : d.index;
                updateHighlight();
            });

        const legend = d3.select(legendContainer)
            .append("div")
            .attr("class", "legend");

        const legendItems = legend.selectAll(".legend-item")
            .data(Object.entries(data))
            .enter()
            .append("div")
            .attr("class", "legend-item")
            .on("click", function (_, d, i) {
                selectedIndex = selectedIndex === i ? null : i;
                updateHighlight();
            });

        legendItems.append("span")
            .attr("class", "legend-color")
            .style("background-color", d => color(d[0]));

        legendItems.append("span")
            .attr("class", "legend-text")
            .text(d => `${d[0]}: ${d[1]}`);

        function updateHighlight() {
            slices.style("opacity", (_, i) => (selectedIndex === null || selectedIndex === i ? 1 : 0.3));
            legendItems.style("opacity", (_, i) => (selectedIndex === null || selectedIndex === i ? 1 : 0.3));
        }
    }

    $: if (data && Object.keys(data).length > 0) {
        drawPieChart();
    }
</script>

<div class="chart-container">
    <h3 style="margin-top: 0rem;" class="chart-title">{title}</h3>
    <div bind:this={container} class="chart-svg"></div>
    <div bind:this={legendContainer} class="legend-container"></div>
</div>

<style>
    .chart-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin-bottom: 20px;
        max-width: 400px;
        padding: 0px;
    }

    .chart-title {
        font-size: 18px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 10px;
    }

    .tooltip {
        position: absolute;
        background: rgba(0, 0, 0, 0.75);
        color: #fff;
        padding: 6px 10px;
        border-radius: 5px;
        font-size: 12px;
        pointer-events: none;
        transition: opacity 0.3s ease-in-out;
        z-index: 10;
    }

    .legend-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px;
        margin-top: 0px;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 6px;
        background: #f9f9f9;
        max-width: 250px;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        color: #333;
        font-weight: 500;
        cursor: pointer;
        transition: opacity 0.2s ease-in-out;
    }

    .legend-item:hover {
        opacity: 0.7;
    }

    .legend-color {
        width: 14px;
        height: 14px;
        display: inline-block;
        border-radius: 3px;
    }
</style>
