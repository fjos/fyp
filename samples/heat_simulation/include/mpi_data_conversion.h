#include "mpi_data_abstractions.h"

static MPI_Datatype convert_type(MPI_Data_Abstraction::Data_Type type)
{
    switch (type)
    {
        case MPI_Data_Abstraction::type_unknown: throw std::runtime_error("MPI_Datatype convert_type(MPI_Data_Abstraction::Data_Type) unkown Data_Type");
        case MPI_Data_Abstraction::type_bool: return MPI::BOOL;
        case MPI_Data_Abstraction::type_char: return MPI::CHAR;
        case MPI_Data_Abstraction::type_unsigned_char: return MPI::UNSIGNED_CHAR;
        case MPI_Data_Abstraction::type_short: return MPI::SHORT;
        case MPI_Data_Abstraction::type_unsigned_short: return MPI::UNSIGNED_SHORT;
        case MPI_Data_Abstraction::type_int: return MPI::INT;
        case MPI_Data_Abstraction::type_unsigned_int: return MPI::UNSIGNED;
        case MPI_Data_Abstraction::type_long: return MPI::LONG;
        case MPI_Data_Abstraction::type_unsigned_long: return MPI::UNSIGNED_LONG;
        case MPI_Data_Abstraction::type_float: return MPI::FLOAT;
        case MPI_Data_Abstraction::type_double: return MPI::DOUBLE;
    };
    throw std::runtime_error("MPI_Data_Type convert_type(MPI_Data_Abstraction::Data_Type) failed");
}