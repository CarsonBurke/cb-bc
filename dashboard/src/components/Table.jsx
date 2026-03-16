export function Table({ children, className = '' }) {
  return (
    <div className="overflow-x-auto">
      <table className={`w-full border-collapse text-[13px] ${className}`}>
        {children}
      </table>
    </div>
  )
}

export function Th({ children, className = '' }) {
  return (
    <th className={`py-2 px-3 text-left text-[12px] text-subtext0 font-medium ${className}`}>
      {children}
    </th>
  )
}

export function Td({ children, className = '' }) {
  return (
    <td className={`py-2.5 px-3 ${className}`}>
      {children}
    </td>
  )
}

export function Thead({ children }) {
  return <thead><tr className="border-b border-surface0">{children}</tr></thead>
}

export function Tr({ children, className = '', onClick }) {
  return (
    <tr
      className={`border-b border-surface0/50 transition-colors hover:bg-surface0/40 ${className}`}
      onClick={onClick}
    >
      {children}
    </tr>
  )
}
