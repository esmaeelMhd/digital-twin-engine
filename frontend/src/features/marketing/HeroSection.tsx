import clsx from 'clsx';
import { useEffect, useRef } from 'react';
import {
  Activity,
  ArrowRight,
  BadgeCheck,
  Cpu,
  FlaskConical,
  Gauge,
  ShieldCheck,
} from 'lucide-react';

import type { DemoPageResponse } from '../../api/types';

import styles from './marketing.module.css';

type HeroSectionProps = {
  page: DemoPageResponse;
};

const chips = [
  { label: 'Plant adaptation', icon: Cpu },
  { label: 'Scenario compare', icon: Activity },
  { label: 'Constraint risk', icon: ShieldCheck },
  { label: 'Stabilization plans', icon: Gauge },
];

export function HeroSection({ page }: HeroSectionProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stats = [
    { value: String(page.demos.length), label: 'Live unit workflows', icon: FlaskConical },
    {
      value: String(page.release.runtime_samples),
      label: 'Samples per decision run',
      icon: ShieldCheck,
    },
    {
      value: page.release.runtime_loaded ? 'Loaded' : 'Fallback',
      label: 'Shared runtime',
      icon: Cpu,
    },
    {
      value: '1 unit',
      label: 'Recommended first scope',
      icon: BadgeCheck,
    },
  ];

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
          <span className="pill">Industrial decision support for one critical unit</span>
          <h1 className={styles.heroTitle}>{page.product_name}</h1>
          <p className={styles.heroLead}>{page.headline}</p>
          <p className={styles.heroBody}>
            {page.summary}
          </p>
          <div className={styles.heroActions}>
            <a className={clsx('button-primary', styles.heroButton)} href="#demo">
              See the live pilot
              <ArrowRight size={16} aria-hidden="true" />
            </a>
            <a className={clsx('button-secondary', styles.heroButton)} href="#proof">
              Review the proof
            </a>
          </div>
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
