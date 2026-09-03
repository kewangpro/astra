// The API serialises timestamps as naive ISO (no trailing "Z" or offset) but
// stores them in UTC. `new Date("2026-09-03T01:34:26")` would be read as *local*
// time, so every displayed timestamp would be off by the viewer's UTC offset.
// Normalise here: append "Z" unless the string already carries a zone marker.
export function parseTs(iso: string): Date {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
}
