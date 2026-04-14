import { useEffect, useRef } from 'react';
import {
  Activity,
  Cpu,
  Factory,
  FlaskConical,
  Gauge,
  Orbit,
  ShieldCheck,
} from 'lucide-react';

import type { DemoPageResponse } from '../../api/types';

import styles from './marketing.module.css';

type HeroSectionProps = {
  page: DemoPageResponse;
};

const stats = [
  { value: '10+', label: 'Industries', icon: Factory },
  { value: '3', label: 'Proven Systems', icon: FlaskConical },
  { value: '< 5 min', label: 'Adaptation Time', icon: Gauge },
  { value: '24/7', label: 'Real-Time API', icon: ShieldCheck },
];

const chips = [
  { label: 'Neural SDE', icon: Orbit },
  { label: 'Few-Shot Transfer', icon: Cpu },
  { label: 'Live Monitoring', icon: Activity },
  { label: 'Uncertainty Quantified', icon: ShieldCheck },
];

export function HeroSection({ page }: HeroSectionProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return undefined;
    }

    const particleCount = 65;
    const particles = Array.from({ length: particleCount }, () => ({
      x: Math.random(),
      y: Math.random(),
      vx: (Math.random() - 0.5) * 0.00016,
      vy: (Math.random() - 0.5) * 0.00016,
      radius: Math.random() * 1.8 + 0.4,
    }));

    let width = 0;
    let height = 0;
    let frame = 0;

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    };

    const render = () => {
      context.clearRect(0, 0, width, height);
      for (const particle of particles) {
        particle.x += particle.vx;
        particle.y += particle.vy;
        if (particle.x <= 0 || particle.x >= 1) {
          particle.vx *= -1;
        }
        if (particle.y <= 0 || particle.y >= 1) {
          particle.vy *= -1;
        }
      }

      for (let index = 0; index < particles.length; index += 1) {
        const source = particles[index];
        for (let compareIndex = index + 1; compareIndex < particles.length; compareIndex += 1) {
          const target = particles[compareIndex];
          const dx = source.x - target.x;
          const dy = source.y - target.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < 0.18) {
            context.strokeStyle = `rgba(250, 247, 240, ${0.18 - distance})`;
            context.lineWidth = 0.5;
            context.beginPath();
            context.moveTo(source.x * width, source.y * height);
            context.lineTo(target.x * width, target.y * height);
            context.stroke();
          }
        }
      }

      for (const particle of particles) {
        context.fillStyle = 'rgba(250, 247, 240, 0.32)';
        context.beginPath();
        context.arc(particle.x * width, particle.y * height, particle.radius, 0, Math.PI * 2);
        context.fill();
      }

      frame = window.requestAnimationFrame(render);
    };

    resize();
    render();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      window.cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <>
      <section className={styles.hero}>
        <canvas ref={canvasRef} className={styles.heroCanvas} />
        <div className={styles.heroInner}>
          <span className="pill">Physics-Informed AI for Industry</span>
          <h1 className={styles.heroTitle}>
            Your Plant
            <br />
            Deserves
            <br />a <em>Brain</em>
          </h1>
          <p className={styles.heroBody}>
            {page.product_name} turns historian data into uncertainty-aware forecasts and
            control recommendations without abandoning the physics of the process.
          </p>
          <div className={styles.heroChips}>
            {chips.map((chip) => (
              <span key={chip.label} className="pill">
                <chip.icon size={14} aria-hidden="true" />
                {chip.label}
              </span>
            ))}
          </div>
        </div>
      </section>

      <div className={styles.statsShell}>
        <div className={styles.statsGrid}>
          {stats.map((item) => (
            <article key={item.label} className={styles.statCard}>
              <div className={styles.statHead}>
                <span className={styles.statLabel}>{item.label}</span>
                <item.icon className={styles.statIcon} size={18} aria-hidden="true" />
              </div>
              <p className={styles.statValue}>{item.value}</p>
            </article>
          ))}
        </div>
      </div>
    </>
  );
}
