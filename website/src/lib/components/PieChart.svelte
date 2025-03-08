<script>
    import * as d3 from 'd3';

    export let data = [];
    export let title = '';
    export let selectedIndex = -1;
    export let colors = d3.scaleOrdinal(d3.schemeTableau10);
    export let transitionDuration = 500;

    let pieData;
    let sliceGenerator = d3.pie().value((d) => d.value).sort(null);
    let arcGenerator = d3.arc().innerRadius(0).outerRadius(50);

    let wedges = {};
    let oldData = [];

    $: {
        oldData = pieData;
        pieData = d3.sort(data.map((d) => ({ ...d })), (d) => d.label);

        let arcData = sliceGenerator(pieData);
        let arcs = arcData.map((d) => arcGenerator(d));

        pieData = pieData.map((d, i) => ({
            ...d,
            ...arcData[i],
            arc: arcs[i],
        }));

        transitionArcs();
    }

    function toggleWedge(index, event) {
        if (!event.key || event.key === 'Enter') {
            selectedIndex = selectedIndex === index ? -1 : index;
        }
    }

    function handleMouseEnter(label) {
        pieData.forEach((slice) => {
            const pathEl = wedges[slice.label]; // path element for that label
            pathEl.style.opacity = (slice.label === label) ? '1' : '0.5';
        });
    }

    function handleMouseLeave() {
        Object.values(wedges).forEach((pathEl) => {
            pathEl.style.opacity = '1';
        });
    }

    function transitionArcs() {
        let wedgeElements = Object.values(wedges);

        d3.selectAll(wedgeElements)
            .transition('arc')
            .duration(transitionDuration)
            .styleTween('d', function (_, index) {
                let label = Object.keys(wedges)[index];
                let d = pieData.find((d) => d.label === label);
                let d_old = oldData.find((d) => d.label === label);

                if (!d || !d_old) {
                    return;
                }

                let from = { ...d_old };
                let to = { ...d };
                let angleInterpolator = d3.interpolate(from, to);

                return (t) => {
                    return `path("${arcGenerator(angleInterpolator(t))}")`;
                };
            });
    }

    function transitionArc(wedge, label) {
        label ??= Object.entries(wedges).find(([lbl, w]) => w === wedge)?.[0];
        if (!label) return null;

        let d = pieData.find((d) => d.label === label);
        let d_old = oldData.find((d) => d.label === label);

        let from = d_old ? { ...d_old } : getEmptyArc(label, oldData);
        let to = d ? { ...d } : getEmptyArc(label, pieData);

        if (sameArc(from, to)) return null;

        let angleInterpolator = d3.interpolate(from, to);

        return {
            d,
            d_old,
            from,
            to,
            interpolator: (t) => `path("${arcGenerator(angleInterpolator(t))}")`,
            type: d ? (d_old ? "update" : "in") : "out",
        };
    }

    function sameArc(arc1, arc2) {
        if (!arc1 && !arc2) return true;
        if (!arc1 || !arc2) return false;
        return arc1.startAngle === arc2.startAngle && arc1.endAngle === arc2.endAngle;
    }

    function arc(wedge) {
        let transition = transitionArc(wedge);
        if (!transition) return { duration: 0, css: "" };

        return {
            duration: transitionDuration,
            easing: d3.easeCubic,
            css: (t, u) =>
                `d: ${transition.interpolator(
                    transition.type === "out" ? u : t
                )}`,
        };
    }

    function getEmptyArc(label, data = pieData) {
        let labels = d3.sort(new Set([...oldData, ...pieData].map((d) => d.label)));
        let labelIndex = labels.indexOf(label);
        let sibling;
        for (let i = labelIndex - 1; i >= 0; i--) {
            sibling = data.find((d) => d.label === labels[i]);
            if (sibling) break;
        }
        let angle = sibling?.endAngle ?? 0;
        return { startAngle: angle, endAngle: angle };
    }

</script>

<div class="container">
    {#if title}
      <h3 class="chart-title">{title}</h3>
    {/if}

    <svg viewBox="-50 -50 100 100">
        {#each pieData as d, i (d.label)}
            <path 
                bind:this={wedges[d.label]}
                transition:arc
                d={d.arc}
                fill={colors(d.label)}
                class:selected={selectedIndex === i}
                tabindex="0"
                role="button"
                aria-label="Select wedge {i}"
                on:click={(e) => toggleWedge(i, e)}
                on:keyup={(e) => toggleWedge(i, e)}
                on:mouseenter={() => handleMouseEnter(d.label)}
                on:mouseleave={handleMouseLeave}
                on:focus={() => handleMouseEnter(d.label)}
                on:blur={handleMouseLeave}
                style="
                    --start-angle: { d.startAngle }rad; 
                    --end-angle: { d.endAngle }rad;
                "
            />
        {/each}
    </svg>

    <ul class="legend">
        {#each pieData as d, i}
        <li 
            class:selected={selectedIndex === i}
            style="--color: {colors(d.id ?? d.label)}"
            on:click={(e) => toggleWedge(i, e)}
        >
            <span class="swatch"></span>
            {d.label} <em>({d.value})</em>
        </li>
        {/each}
    </ul>
</div>

<style>
    .container {
        align-items: center;
        text-align: center;
        gap: 2em;
        max-width: 100%;
        margin: 1em auto;
        padding: 0em;
        min-width: 5em;
    }

    svg {
        max-width: 20em;
        height: auto;
        min-width: 10em;
        margin-block: 1em;
        overflow: visible;
        min-width: 15em;
        margin-bottom: 2em;
    }

    svg:has(path:hover, path:focus-visible) {
        path:not(:hover, :focus-visible) {
            opacity: 50%;
        }
    }

    path {
        transition: transform, opacity, fill 300ms;
        transition-property: transform, opacity, fill;
        cursor: pointer;
        outline: none;
        --angle: calc(var(--end-angle) - var(--start-angle));
        --mid-angle: calc(var(--start-angle) + var(--angle) / 2);
        transform: rotate(var(--mid-angle)) translateY(0) scale(1) rotate(calc(-1 * var(--mid-angle)));

        &.selected {
            transform: rotate(var(--mid-angle)) translateY(-6px) scale(1.1) rotate(
                calc(-1 * var(--mid-angle))
            );
        }
    }

    path:focus-visible {
        outline: 2px solid var(--color);
        outline-offset: 2px;
    }

    .selected {
        --color: oklch(87.35% 0.0839 11.65) !important;

        &:is(path) {
            fill: var(--color);
        }
    }

    .chart-title {
        font-size: 18px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 20px;
    }

    .legend {
        flex: 1;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(9em, 1fr));
        padding: 1em;
        margin: 1em;
        border: 1px solid #ccc;
        border-radius: 8px;
        background-color: #f9f9f9;
        min-width: 8em;
        margin-left: 1em;
    }
    .legend li {
        display: flex;
        align-items: center;
        gap: 0.5em;
        padding: 0.2em;
        border-radius: 4px;
    }
    .swatch {
        display: inline-block;
        width: 1em;
        height: 1em;
        aspect-ratio: 1 / 1;
        background-color: var(--color);
        border-radius: 50%;
    }
    .legend li em {
        font-style: italic;
        color: #666;
    }
</style>