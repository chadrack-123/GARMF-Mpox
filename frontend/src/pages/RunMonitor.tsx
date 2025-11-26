import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Container,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
} from '@mui/material'
import { Visibility as VisibilityIcon } from '@mui/icons-material'
import { getRuns } from '../api/client'
import type { Run } from '../types'

export default function RunMonitor() {
  const navigate = useNavigate()

  const { data: runs } = useQuery({
    queryKey: ['runs'],
    queryFn: async () => {
      const response = await getRuns()
      return response.data as Run[]
    },
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success'
      case 'running': return 'info'
      case 'failed': return 'error'
      default: return 'default'
    }
  }

  return (
    <Container maxWidth="xl">
      <Typography variant="h4" gutterBottom>
        Run Monitor
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Run ID</TableCell>
              <TableCell>Study ID</TableCell>
              <TableCell>Kind</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Started</TableCell>
              <TableCell>Finished</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {runs?.map((run) => (
              <TableRow key={run.id}>
                <TableCell>{run.id}</TableCell>
                <TableCell>{run.study_id}</TableCell>
                <TableCell>{run.kind}</TableCell>
                <TableCell>
                  <Chip
                    label={run.status}
                    color={getStatusColor(run.status) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  {run.started_at ? new Date(run.started_at).toLocaleString() : '-'}
                </TableCell>
                <TableCell>
                  {run.finished_at ? new Date(run.finished_at).toLocaleString() : '-'}
                </TableCell>
                <TableCell>
                  <Button
                    size="small"
                    startIcon={<VisibilityIcon />}
                    onClick={() => navigate(`/runs/${run.id}/xai`)}
                  >
                    View
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Container>
  )
}
