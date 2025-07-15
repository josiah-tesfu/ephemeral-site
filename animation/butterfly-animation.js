const canvas = document.getElementById('dotCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const all_coords = [
      ...updated_coords.map(p => [...p, "main"]),
      ...ephemeral_coords.map(p => [...p, "ephemeral"])
    ];

    function getBounds(data) {
      return data.reduce((acc, [x, y]) => {
        acc.minX = Math.min(acc.minX, x);
        acc.minY = Math.min(acc.minY, y);
        acc.maxX = Math.max(acc.maxX, x);
        acc.maxY = Math.max(acc.maxY, y);
        return acc;
      }, { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity });
    }

    const boundsMain = getBounds(updated_coords);
    const boundsEph = getBounds(ephemeral_coords);

    const baseScale = Math.min(
      canvas.width / (boundsMain.maxX - boundsMain.minX),
      canvas.height / (boundsMain.maxY - boundsMain.minY)
    );

    const globalScale = baseScale * 0.8; // zoomed out effect

    const offsetMainX = (canvas.width - globalScale * (boundsMain.maxX - boundsMain.minX)) / 2;
    const offsetMainY = (canvas.height - globalScale * (boundsMain.maxY - boundsMain.minY)) / 2 + 100;

    const offsetEphX = (canvas.width - globalScale * (boundsEph.maxX - boundsEph.minX)) / 2;
    const offsetEphY = (canvas.height - globalScale * (boundsEph.maxY - boundsEph.minY)) / 2 - 265;

    const baseButterflySize = 20;
    const butterflySize = baseButterflySize * globalScale / baseScale;

    const imageCache = {};
    const dots = all_coords.map(([x, y, hex, source]) => {
      const { baseX, baseY } = (() => {
        if (source === "ephemeral") {
          const bx = offsetEphX + (x - boundsEph.minX) * globalScale;
          const by = offsetEphY + (y - boundsEph.minY) * globalScale;
          return { baseX: bx, baseY: by };
        } else {
          const bx = offsetMainX + (x - boundsMain.minX) * globalScale;
          const by = offsetMainY + (y - boundsMain.minY) * globalScale;
          return { baseX: bx, baseY: by };
        }
      })();

      const motion = (source === "ephemeral")
        ? { amplitude: 2, speed: 0.004, dispersal: 800 }
        : { amplitude: 4, speed: 0.004, dispersal: 800 };

      const a = Math.random() * 2 * Math.PI;
      const r = Math.random() * motion.dispersal;
      const targetX = baseX + Math.cos(a) * r;
      const targetY = baseY + Math.sin(a) * r;

      const key = Object.entries({
        'white': '#FFFFFF', 'pale_ivory': '#F2E8D5', 'sky_blue': '#B8D9F9',
        'warm_orange': '#F6A06D', 'lavender': '#9E85C4', 'muted_green': '#6BBF59',
        'warm_brown': '#523F3A', 'slate_blue': '#303952',
        'bright_red': '#FA5252', 'vivid_yellow': '#FEE440'
      }).find(([, val]) => val.toLowerCase() === hex.toLowerCase())?.[0] || 'white';

      if (!imageCache[key]) {
        const img = new Image();
        img.src = `butterfly_colored/butterfly_${key}.png`;
        imageCache[key] = img;
      }

      return {
        baseX, baseY, targetX, targetY,
        phaseX: Math.random() * 1000,
        phaseY: Math.random() * 1000,
        image: imageCache[key],
        rotation: Math.random() * 2 * Math.PI,
        source,
        ...motion
      };
    });

    const cycleDuration = 15000;
    const holdDuration = 3000;

    function easeInOutCubic(t) {
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    function animate(timestamp) {
      const totalCycle = cycleDuration + holdDuration;
      const t = timestamp % totalCycle;
      let animT = (t < holdDuration) ? 0 : (() => {
        const phase = (t - holdDuration) / cycleDuration;
        return phase < 0.5 ? easeInOutCubic(phase * 2) : easeInOutCubic(2 - phase * 2);
      })();

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const dot of dots) {
        const jitterX = Math.sin(timestamp * dot.speed + dot.phaseX) * dot.amplitude;
        const jitterY = Math.cos(timestamp * dot.speed + dot.phaseY) * dot.amplitude;

        const interpX = dot.baseX + (dot.targetX - dot.baseX) * animT + jitterX;
        const interpY = dot.baseY + (dot.targetY - dot.baseY) * animT + jitterY;

        ctx.save();
        ctx.translate(interpX, interpY);
        ctx.rotate(dot.rotation);
        ctx.drawImage(dot.image, -butterflySize / 2, -butterflySize / 2, butterflySize, butterflySize);
        ctx.restore();
      }
      requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);