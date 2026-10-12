export const DRINKS = [
  {
    id: "blue-hint-water",
    name: "Blue Hint Water",
    flavor: "Blackberry",
    detectInstruction: "middle of blue drink can",
    color: "#3478c9",
  },
  {
    id: "red-hint-water",
    name: "Red Hint Water",
    flavor: "Lemon",
    detectInstruction: "middle of red drink can",
    color: "#c84a43",
  },
  {
    id: "monster-energy",
    name: "Monster Energy Drink",
    flavor: "Energy drink",
    detectInstruction: "black drink can",
    color: "#3d8847",
  },
];

export function findDrink(id) {
  return DRINKS.find((drink) => drink.id === id);
}

export function isDeliveryWaypoint(name, waypoints) {
  return Boolean(name && name !== "home" && name !== "snack1" && waypoints.includes(name));
}
