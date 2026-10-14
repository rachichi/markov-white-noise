import podium from "./assets/podium.png";
import type { SpeakerPage } from "./speakers";

const mod = (n: number, m: number) => ((n % m) + m) % m;

export function Carousel({ items, index, onChange }: { items: SpeakerPage[]; index: number; onChange: (i: number) => void }) {
  const n = items.length;
  const prev = mod(index - 1, n);
  const next = mod(index + 1, n);
  const slot = (i: number) => (i === index ? "center" : i === prev ? "left" : i === next ? "right" : "hidden");

  return (
    <div className="carousel" role="group" aria-roledescription="carousel" aria-label="Speakers">
      <button type="button" className="carousel__arrow carousel__arrow--left" onClick={() => onChange(prev)}
        aria-label={`Previous: ${items[prev].name}`}>
        ‹
      </button>
      <div className="carousel__stage">
        {items.map((s, i) => (
          <button
            key={s.slug}
            type="button"
            className={`carousel__item carousel__item--${slot(i)}`}
            onClick={() => onChange(i)}
            aria-label={s.name}
            aria-current={i === index}
            tabIndex={i === index ? -1 : 0}
          >
            <img
              src={s.portrait}
              alt={`Portrait of ${s.name}`}
              style={s.bottomPadding ? { transform: `translateY(${s.bottomPadding * 100}%)` } : undefined}
            />
          </button>
        ))}
        <img className="carousel__podium" src={podium} alt="" aria-hidden="true" />
      </div>
      <button type="button" className="carousel__arrow carousel__arrow--right" onClick={() => onChange(next)}
        aria-label={`Next: ${items[next].name}`}>
        ›
      </button>
    </div>
  );
}
